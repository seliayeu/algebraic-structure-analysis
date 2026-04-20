#include "Transform/Operators/DIASoftmaxRewrite.h"

#include <cstdint>

#include "Analysis/BandedStructureAnalysis.h"
#include "Dialect/DIA/DIAOps.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace mlir::bpa {

struct DIASoftmaxPattern : public OpRewritePattern<dia::SoftmaxOp> {
    using OpRewritePattern::OpRewritePattern;

    Value buildInBandPredicate(OpBuilder& b, Location loc, Value i, Value j, Value constL,
                               Value constU) const {
        // inLower: j >= i - L   <=>  j >= i - L
        Value iMinusL = arith::SubIOp::create(b, loc, i, constL);
        Value inLower = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sge, j, iMinusL);
        // inUpper: j <= i + U
        Value iPlusU = arith::AddIOp::create(b, loc, i, constU);
        Value inUpper = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::sle, j, iPlusU);
        return arith::AndIOp::create(b, loc, inLower, inUpper);
    }

    // -----------------------------------------------------------------------
    // Main lowering
    // -----------------------------------------------------------------------
    LogicalResult denseBandedSoftmaxToLinalg(dia::SoftmaxOp op, PatternRewriter& rewriter,
                                             const BandedSubMatrix& bandInfo) const {
        Location loc = op.getLoc();
        MLIRContext* ctx = rewriter.getContext();
        FloatType f32 = rewriter.getF32Type();

        Value input = op.getInput();
        auto inputType = cast<RankedTensorType>(input.getType());

        assert(inputType.getRank() == 2 && "expected rank-2 input");
        const int64_t N = inputType.getDimSize(0);
        assert(inputType.getDimSize(1) == N && "expected square matrix");

        const int64_t L = bandInfo.Property.LowerBandwidth;
        const int64_t U = bandInfo.Property.UpperBandwidth;

        auto tensorNNType = RankedTensorType::get({ N, N }, f32);
        auto tensorNType = RankedTensorType::get({ N }, f32);
        auto tensorScalar = RankedTensorType::get({}, f32);

        Value constL = arith::ConstantIndexOp::create(rewriter, loc, L);
        Value constU = arith::ConstantIndexOp::create(rewriter, loc, U);
        Value negInfF =
            arith::ConstantFloatOp::create(rewriter, loc, f32,
                                           llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(),
                                                                 /*Negative=*/true));
        Value zeroF = arith::ConstantFloatOp::create(rewriter, loc, f32, llvm::APFloat(0.0f));

        // ---- Affine maps -------------------------------------------------------
        // (d0, d1) -> (d0, d1)   full 2-D identity
        AffineMap map2D = AffineMap::getMultiDimIdentityMap(2, ctx);
        // (d0, d1) -> (d0)       row reduction / row broadcast
        AffineMap mapRow = AffineMap::get(2, 0, { rewriter.getAffineDimExpr(0) }, ctx);

        using IT = utils::IteratorType;
        SmallVector<IT> rowReduction = { IT::parallel, IT::reduction };
        SmallVector<IT> allParallel = { IT::parallel, IT::parallel };

        // ======================================================================
        // Pass 1 – per-row max over the band
        //   input  (d0,d1) -> (d0,d1)   [parallel × reduction]
        //   output (d0,d1) -> (d0)      scalar per row
        // Out-of-band positions contribute -inf so they never win the max.
        // ======================================================================
        Value maxInit =
            linalg::FillOp::create(
                rewriter, loc, ValueRange{ negInfF },
                ValueRange{ tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{ N }, f32) })
                .getResult(0);

        Value rowMax =
            linalg::GenericOp::create(
                rewriter, loc, TypeRange{ tensorNType }, ValueRange{ input }, ValueRange{ maxInit },
                ArrayRef<AffineMap>{ map2D, mapRow }, rowReduction,
                [&](OpBuilder& b, Location loc, ValueRange args) {
                    // args[0] = input[i,j],  args[1] = running max for row i
                    Value i = linalg::IndexOp::create(b, loc, 0);
                    Value j = linalg::IndexOp::create(b, loc, 1);
                    Value inBand = buildInBandPredicate(b, loc, i, j, constL, constU);
                    Value effective = arith::SelectOp::create(b, loc, inBand, args[0], negInfF);
                    Value newMax = arith::MaximumFOp::create(b, loc, effective, args[1]);
                    linalg::YieldOp::create(b, loc, newMax);
                })
                .getResult(0);

        // ======================================================================
        // Pass 2 – exp(x[i,j] - rowMax[i]),  zero for out-of-band positions
        //   input   (d0,d1) -> (d0,d1)   [parallel × parallel]
        //   rowMax  (d0,d1) -> (d0)      broadcast across columns
        //   output  (d0,d1) -> (d0,d1)
        // ======================================================================
        Value expInit = tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{ N, N }, f32);

        Value exps =
            linalg::GenericOp::create(
                rewriter, loc, TypeRange{ tensorNNType }, ValueRange{ input, rowMax },
                ValueRange{ expInit }, ArrayRef<AffineMap>{ map2D, mapRow, map2D }, allParallel,
                [&](OpBuilder& b, Location loc, ValueRange args) {
                    // args[0] = input[i,j],  args[1] = rowMax[i]
                    Value i = linalg::IndexOp::create(b, loc, 0);
                    Value j = linalg::IndexOp::create(b, loc, 1);
                    Value inBand = buildInBandPredicate(b, loc, i, j, constL, constU);
                    Value shifted = arith::SubFOp::create(b, loc, args[0], args[1]);
                    Value expVal = math::ExpOp::create(b, loc, shifted);
                    // softmax(-inf) = 0: zero out-of-band elements explicitly
                    Value result = arith::SelectOp::create(b, loc, inBand, expVal, zeroF);
                    linalg::YieldOp::create(b, loc, result);
                })
                .getResult(0);

        // ======================================================================
        // Pass 3 – per-row sum of exps
        //   exps   (d0,d1) -> (d0,d1)   [parallel × reduction]
        //   output (d0,d1) -> (d0)
        // ======================================================================
        Value sumInit =
            linalg::FillOp::create(
                rewriter, loc, ValueRange{ zeroF },
                ValueRange{ tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{ N }, f32) })
                .getResult(0);

        Value rowSum =
            linalg::GenericOp::create(
                rewriter, loc, TypeRange{ tensorNType }, ValueRange{ exps }, ValueRange{ sumInit },
                ArrayRef<AffineMap>{ map2D, mapRow }, rowReduction,
                [&](OpBuilder& b, Location loc, ValueRange args) {
                    // args[0] = exps[i,j],  args[1] = running sum for row i
                    Value newSum = arith::AddFOp::create(b, loc, args[0], args[1]);
                    linalg::YieldOp::create(b, loc, newSum);
                })
                .getResult(0);

        // ======================================================================
        // Pass 4 – normalize: out[i,j] = exps[i,j] / rowSum[i]
        //   exps   (d0,d1) -> (d0,d1)   [parallel × parallel]
        //   rowSum (d0,d1) -> (d0)      broadcast across columns
        //   output (d0,d1) -> (d0,d1)
        //
        // Out-of-band positions already hold 0.0 from pass 2, so dividing
        // them by rowSum still gives 0.0 — no extra band check needed here.
        // ======================================================================
        Value outInit = tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{ N, N }, f32);

        Value result =
            linalg::GenericOp::create(
                rewriter, loc, TypeRange{ tensorNNType }, ValueRange{ exps, rowSum },
                ValueRange{ outInit }, ArrayRef<AffineMap>{ map2D, mapRow, map2D }, allParallel,
                [&](OpBuilder& b, Location loc, ValueRange args) {
                    // args[0] = exps[i,j],  args[1] = rowSum[i]
                    Value normalized = arith::DivFOp::create(b, loc, args[0], args[1]);
                    linalg::YieldOp::create(b, loc, normalized);
                })
                .getResult(0);

        rewriter.replaceOp(op, result);
        return success();
    }

    LogicalResult matchAndRewrite(dia::SoftmaxOp op, PatternRewriter& rewriter) const {
        auto dict = op->getAttrDictionary();
        if (!dict) dict = DictionaryAttr();

        const BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);
        if (!opBandInfo.IsDia) return denseBandedSoftmaxToLinalg(op, rewriter, opBandInfo);

        return failure();
    }
};

void addDIASoftmaxPatterns(RewritePatternSet& patterns) {
    patterns.add<DIASoftmaxPattern>(patterns.getContext());
}

}  // namespace mlir::bpa
//
