#include "Transform/DenseSoftmaxRewrite.h"

#include <cstdint>

#include "Utils/TransformUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bpa {

#define GEN_PASS_DEF_DENSESOFTMAXREWRITE
#include "lib/Transform/Passes.h.inc"

struct DenseSoftmaxPattern : public OpRewritePattern<dia::SoftmaxOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(dia::SoftmaxOp op, PatternRewriter& rewriter) const {
        Location loc = op.getLoc();
        MLIRContext* ctx = rewriter.getContext();
        FloatType f32 = rewriter.getF32Type();

        Value input = op.getInput();
        auto inputType = cast<RankedTensorType>(input.getType());
        assert(inputType.getRank() == 2 && "expected rank-2 input");

        const int64_t rows = inputType.getDimSize(0);
        const int64_t cols = inputType.getDimSize(1);

        // ---- constants -------------------------------------------------------
        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value dimRows = arith::ConstantIndexOp::create(rewriter, loc, rows);

        Value negInfF =
            arith::ConstantFloatOp::create(rewriter, loc, f32,
                                           llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(),
                                                                 /*Negative=*/true));
        Value zeroF = arith::ConstantFloatOp::create(rewriter, loc, f32, llvm::APFloat(0.0f));

        // Use OpFoldResults for sizes/offsets to explicitly mark them as static
        // to MLIR, avoiding the "expected rank-reduced version" type mismatch.
        OpFoldResult staticZero = rewriter.getIndexAttr(0);
        OpFoldResult staticOne = rewriter.getIndexAttr(1);
        OpFoldResult staticStride = rewriter.getIndexAttr(1);
        OpFoldResult colSizeOFR = rewriter.getIndexAttr(cols);

        Value outEmpty =
            tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{ rows, cols }, f32);
        Value outInit =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zeroF }, ValueRange{ outEmpty })
                .getResult(0);

        // ---- Pre-allocate buffers for reuse across iterations ----------------
        Value expBuffer = tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{ cols }, f32);
        Value normBuffer = tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{ cols }, f32);

        AffineMap map1D = AffineMap::getMultiDimIdentityMap(1, ctx);
        AffineMap mapSc = AffineMap::get(1, 0, ctx);

        SmallVector<utils::IteratorType> red1D = { utils::IteratorType::reduction };
        SmallVector<utils::IteratorType> par1D = { utils::IteratorType::parallel };

        auto sliceType = RankedTensorType::get({ cols }, f32);

        auto rowLoop = scf::ForOp::create(rewriter, loc, c0, dimRows, c1, ValueRange{ outInit });
        rewriter.setInsertionPointToStart(rowLoop.getBody());
        {
            Value i = rowLoop.getInductionVar();
            Value outCarried = rowLoop.getRegionIterArg(0);

            // ---- rank-reducing extract_slice (entire row) --------------------
            SmallVector<OpFoldResult> extractOffsets = { getAsOpFoldResult(i), staticZero };
            SmallVector<OpFoldResult> extractSizes = { staticOne, colSizeOFR };
            SmallVector<OpFoldResult> extractStrides = { staticStride, staticStride };

            Value slice = tensor::ExtractSliceOp::create(
                rewriter, loc, sliceType, input, extractOffsets, extractSizes, extractStrides);

            // ==================================================================
            // Step 1 – row max (reduce over the entire row)
            // ==================================================================
            Value maxInitTensor =
                linalg::FillOp::create(
                    rewriter, loc, ValueRange{ negInfF },
                    ValueRange{ tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{}, f32) })
                    .getResult(0);

            Value rowMaxTensor =
                linalg::GenericOp::create(
                    rewriter, loc, TypeRange{ maxInitTensor.getType() }, ValueRange{ slice },
                    ValueRange{ maxInitTensor }, ArrayRef<AffineMap>{ map1D, mapSc }, red1D,
                    [&](OpBuilder& b, Location loc, ValueRange args) {
                        Value newMax = arith::MaximumFOp::create(b, loc, args[0], args[1]);
                        linalg::YieldOp::create(b, loc, newMax);
                    })
                    .getResult(0);

            // ==================================================================
            // Step 2 – exp(slice[j] - rowMax)
            // ==================================================================
            Value exps = linalg::GenericOp::create(
                             rewriter, loc, TypeRange{ expBuffer.getType() },
                             ValueRange{ slice, rowMaxTensor }, ValueRange{ expBuffer },
                             ArrayRef<AffineMap>{ map1D, mapSc, map1D }, par1D,
                             [&](OpBuilder& b, Location loc, ValueRange args) {
                                 Value shifted = arith::SubFOp::create(b, loc, args[0], args[1]);
                                 Value expVal = math::ExpOp::create(b, loc, shifted);
                                 linalg::YieldOp::create(b, loc, expVal);
                             })
                             .getResult(0);

            // ==================================================================
            // Step 3 – row sum (reduce over exps)
            // ==================================================================
            Value sumInitTensor =
                linalg::FillOp::create(
                    rewriter, loc, ValueRange{ zeroF },
                    ValueRange{ tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{}, f32) })
                    .getResult(0);

            Value rowSumTensor =
                linalg::GenericOp::create(
                    rewriter, loc, TypeRange{ sumInitTensor.getType() }, ValueRange{ exps },
                    ValueRange{ sumInitTensor }, ArrayRef<AffineMap>{ map1D, mapSc }, red1D,
                    [&](OpBuilder& b, Location loc, ValueRange args) {
                        Value newSum = arith::AddFOp::create(b, loc, args[0], args[1]);
                        linalg::YieldOp::create(b, loc, newSum);
                    })
                    .getResult(0);

            // ==================================================================
            // Step 4 – normalize: exps[j] / rowSum
            // ==================================================================
            Value normalized = linalg::GenericOp::create(
                                   rewriter, loc, TypeRange{ normBuffer.getType() },
                                   ValueRange{ exps, rowSumTensor }, ValueRange{ normBuffer },
                                   ArrayRef<AffineMap>{ map1D, mapSc, map1D }, par1D,
                                   [&](OpBuilder& b, Location loc, ValueRange args) {
                                       Value div = arith::DivFOp::create(b, loc, args[0], args[1]);
                                       linalg::YieldOp::create(b, loc, div);
                                   })
                                   .getResult(0);

            // ==================================================================
            // Step 5 – rank-expanding insert_slice back into the output tensor
            // ==================================================================
            SmallVector<OpFoldResult> insertOffsets = { getAsOpFoldResult(i), staticZero };
            SmallVector<OpFoldResult> insertSizes = { staticOne, colSizeOFR };
            SmallVector<OpFoldResult> insertStrides = { staticStride, staticStride };

            Value updatedOut = tensor::InsertSliceOp::create(
                rewriter, loc, normalized, outCarried, insertOffsets, insertSizes, insertStrides);

            scf::YieldOp::create(rewriter, loc, ValueRange{ updatedOut });
        }

        rewriter.replaceOp(op, rowLoop.getResult(0));
        return success();
    }
};

struct DenseSoftmaxRewrite : public impl::DenseSoftmaxRewriteBase<DenseSoftmaxRewrite> {
    using DenseSoftmaxRewriteBase::DenseSoftmaxRewriteBase;

    void runOnOperation() override {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = funcOp.getContext();

        RewritePatternSet patterns(context);
        patterns.add<DenseSoftmaxPattern>(context);

        GreedyRewriteConfig config;
        config.setMaxIterations(1);
        config.setUseTopDownTraversal(true);

        (void)applyPatternsGreedily(funcOp, std::move(patterns), config);
    }
};

}  // namespace mlir::bpa
