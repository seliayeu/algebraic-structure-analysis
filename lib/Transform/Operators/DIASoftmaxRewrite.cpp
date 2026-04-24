#include "Transform/Operators/DIASoftmaxRewrite.h"

#include <cstdint>

#include "Analysis/BandedStructureAnalysis.h"
#include "Dialect/DIA/DIAOps.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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

        // ---- constants -------------------------------------------------------
        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value dimN = arith::ConstantIndexOp::create(rewriter, loc, N);
        Value constL = arith::ConstantIndexOp::create(rewriter, loc, L);
        Value constU = arith::ConstantIndexOp::create(rewriter, loc, U);

        Value negInfF =
            arith::ConstantFloatOp::create(rewriter, loc, f32,
                                           llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(),
                                                                 /*Negative=*/true));
        Value zeroF = arith::ConstantFloatOp::create(rewriter, loc, f32, llvm::APFloat(0.0f));

        // Static OpFoldResults for positions that are always 1.
        // This is what makes rank-reducing extract_slice and rank-expanding
        // insert_slice work: the dropped/added dimension must have a
        // *static* size of 1 (an IntegerAttr), not a dynamic SSA value.
        OpFoldResult staticOne = rewriter.getIndexAttr(1);
        OpFoldResult staticStride = rewriter.getIndexAttr(1);

        Value outEmpty = tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{ N, N }, f32);
        Value outInit =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zeroF }, ValueRange{ outEmpty })
                .getResult(0);

        AffineMap map1D = AffineMap::getMultiDimIdentityMap(1, ctx);
        AffineMap mapSc = AffineMap::get(1, 0, ctx);

        SmallVector<utils::IteratorType> red1D = { utils::IteratorType::reduction };
        SmallVector<utils::IteratorType> par1D = { utils::IteratorType::parallel };
        auto sliceType = RankedTensorType::get({ ShapedType::kDynamic }, f32);

        auto rowLoop = scf::ForOp::create(rewriter, loc, c0, dimN, c1, ValueRange{ outInit });
        rewriter.setInsertionPointToStart(rowLoop.getBody());
        {
            Value i = rowLoop.getInductionVar();
            Value outCarried = rowLoop.getRegionIterArg(0);

            // band bounds: j in [max(0, i-L), min(N, i+U+1))
            Value iMinusL = arith::SubIOp::create(rewriter, loc, i, constL);
            Value jStart = arith::MaxSIOp::create(rewriter, loc, c0, iMinusL);
            Value iPlusUp1 = arith::AddIOp::create(
                rewriter, loc, arith::AddIOp::create(rewriter, loc, i, constU), c1);
            Value jEnd = arith::MinSIOp::create(rewriter, loc, dimN, iPlusUp1);
            Value sliceLen = arith::SubIOp::create(rewriter, loc, jEnd, jStart);

            // ---- rank-reducing extract_slice ---------------------------------
            // offsets = [i,      jStart]
            // sizes   = [1,      sliceLen]   <- dim-0 size is STATIC 1
            // strides = [1,      1]
            //
            // MLIR drops any dimension whose size is a static 1, so the result
            // is tensor<?xf32> (rank 1) instead of tensor<1x?xf32> (rank 2).
            SmallVector<OpFoldResult> extractOffsets = { getAsOpFoldResult(i),
                                                         getAsOpFoldResult(jStart) };
            SmallVector<OpFoldResult> extractSizes = {
                staticOne,                   // static — triggers rank reduction
                getAsOpFoldResult(sliceLen)  // dynamic
            };
            SmallVector<OpFoldResult> extractStrides = { staticStride, staticStride };

            Value slice = tensor::ExtractSliceOp::create(
                rewriter, loc, sliceType, input, extractOffsets, extractSizes, extractStrides);

            // ==================================================================
            // Step 1 – row max  (reduce over the band slice only)
            // input:  slice  (d0)->(d0)  [reduction]
            // output: scalar (d0)->()
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
                        // args[0] = slice[j],  args[1] = running max
                        Value newMax = arith::MaximumFOp::create(b, loc, args[0], args[1]);
                        linalg::YieldOp::create(b, loc, newMax);
                    })
                    .getResult(0);

            // ==================================================================
            // Step 2 – exp(slice[j] - rowMax)  over the band only
            // input:  slice  (d0)->(d0)  [parallel]
            //         rowMax (d0)->()    [broadcast]
            // output: exps   (d0)->(d0)
            // ==================================================================
            Value expEmpty =
                tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{ ShapedType::kDynamic },
                                        f32, ValueRange{ sliceLen });

            Value exps = linalg::GenericOp::create(
                             rewriter, loc, TypeRange{ expEmpty.getType() },
                             ValueRange{ slice, rowMaxTensor }, ValueRange{ expEmpty },
                             ArrayRef<AffineMap>{ map1D, mapSc, map1D }, par1D,
                             [&](OpBuilder& b, Location loc, ValueRange args) {
                                 // args[0] = slice[j],  args[1] = rowMax (broadcast)
                                 Value shifted = arith::SubFOp::create(b, loc, args[0], args[1]);
                                 Value expVal = math::ExpOp::create(b, loc, shifted);
                                 linalg::YieldOp::create(b, loc, expVal);
                             })
                             .getResult(0);

            // ==================================================================
            // Step 3 – row sum  (reduce over exps)
            // input:  exps   (d0)->(d0)  [reduction]
            // output: scalar (d0)->()
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
                        // args[0] = exps[j],  args[1] = running sum
                        Value newSum = arith::AddFOp::create(b, loc, args[0], args[1]);
                        linalg::YieldOp::create(b, loc, newSum);
                    })
                    .getResult(0);

            // ==================================================================
            // Step 4 – normalize: exps[j] / rowSum
            // input:  exps   (d0)->(d0)  [parallel]
            //         rowSum (d0)->()    [broadcast]
            // output: norm   (d0)->(d0)
            // ==================================================================
            Value normEmpty =
                tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{ ShapedType::kDynamic },
                                        f32, ValueRange{ sliceLen });

            Value normalized = linalg::GenericOp::create(
                                   rewriter, loc, TypeRange{ normEmpty.getType() },
                                   ValueRange{ exps, rowSumTensor }, ValueRange{ normEmpty },
                                   ArrayRef<AffineMap>{ map1D, mapSc, map1D }, par1D,
                                   [&](OpBuilder& b, Location loc, ValueRange args) {
                                       // args[0] = exps[j],  args[1] = rowSum (broadcast)
                                       Value div = arith::DivFOp::create(b, loc, args[0], args[1]);
                                       linalg::YieldOp::create(b, loc, div);
                                   })
                                   .getResult(0);

            // ==================================================================
            // Step 5 – rank-expanding insert_slice back into the output tensor
            // source:  normalized  tensor<?xf32>   (rank 1)
            // dest:    outCarried  tensor<NxNxf32>  (rank 2)
            // offsets = [i,      jStart]
            // sizes   = [1,      sliceLen]   <- static 1 re-adds the dimension
            // strides = [1,      1]
            // ==================================================================
            SmallVector<OpFoldResult> insertOffsets = { getAsOpFoldResult(i),
                                                        getAsOpFoldResult(jStart) };
            SmallVector<OpFoldResult> insertSizes = {
                staticOne,                   // static — triggers rank expansion
                getAsOpFoldResult(sliceLen)  // dynamic
            };
            SmallVector<OpFoldResult> insertStrides = { staticStride, staticStride };

            Value updatedOut = tensor::InsertSliceOp::create(
                rewriter, loc, normalized, outCarried, insertOffsets, insertSizes, insertStrides);

            scf::YieldOp::create(rewriter, loc, ValueRange{ updatedOut });
        }

        rowLoop->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, rowLoop.getResult(0));
        return success();
    }

    LogicalResult diaBandedSoftmaxToLinalg(dia::SoftmaxOp op, PatternRewriter& rewriter,
                                           const BandedSubMatrix& bandInfo) const {
        Location loc = op.getLoc();
        MLIRContext* ctx = rewriter.getContext();
        FloatType f32 = rewriter.getF32Type();

        Value input = op.getInput();
        auto inputType = cast<RankedTensorType>(input.getType());

        // DIA storage shape: (L + U + 1) × N
        const int64_t L = bandInfo.Property.LowerBandwidth;
        const int64_t U = bandInfo.Property.UpperBandwidth;
        const int64_t N = inputType.getDimSize(1);  // matrix dimension (columns)

        // ---- constants -----------------------------------------------------------
        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value dimN = arith::ConstantIndexOp::create(rewriter, loc, N);
        Value constL = arith::ConstantIndexOp::create(rewriter, loc, L);
        Value constU = arith::ConstantIndexOp::create(rewriter, loc, U);
        Value constLi64 = arith::ConstantIntOp::create(rewriter, loc, L, 64);

        Value negInfF = arith::ConstantFloatOp::create(
            rewriter, loc, f32,
            llvm::APFloat::getInf(llvm::APFloat::IEEEsingle(), /*Negative=*/true));
        Value zeroF = arith::ConstantFloatOp::create(rewriter, loc, f32, llvm::APFloat(0.0f));

        // Static OpFoldResults for rank reduction/expansion.
        OpFoldResult staticOne = rewriter.getIndexAttr(1);
        OpFoldResult staticStride = rewriter.getIndexAttr(1);

        SmallVector<utils::IteratorType> red1D = { utils::IteratorType::reduction };
        SmallVector<utils::IteratorType> par1D = { utils::IteratorType::parallel };

        // ---- 1-D affine maps reused across all passes --------------------
        AffineMap map1D = AffineMap::getMultiDimIdentityMap(1, ctx);
        AffineMap mapSc = AffineMap::get(1, 0, ctx);  // scalar broadcast
        auto sliceType = RankedTensorType::get({ ShapedType::kDynamic }, f32);

        // ---- outer loop over matrix rows -----------------------------------------
        auto rowLoop = scf::ForOp::create(rewriter, loc, c0, dimN, c1, ValueRange{ input });
        rewriter.setInsertionPointToStart(rowLoop.getBody());
        {
            Value i = rowLoop.getInductionVar();
            Value diaIn = rowLoop.getRegionIterArg(0);

            // Band column bounds for row i: [max(0, i-L), min(N, i+U+1))
            Value iMinusL = arith::SubIOp::create(rewriter, loc, i, constL);
            Value jStart = arith::MaxSIOp::create(rewriter, loc, c0, iMinusL);
            Value iPlusU1 = arith::AddIOp::create(
                rewriter, loc, arith::AddIOp::create(rewriter, loc, i, constU), c1);
            Value jEnd = arith::MinSIOp::create(rewriter, loc, dimN, iPlusU1);
            Value sliceLen = arith::SubIOp::create(rewriter, loc, jEnd, jStart);

            // Calculate starting row in the DIA tensor for this slice
            Value ji64 = arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), jStart);
            Value ii64 = arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(), i);
            Value diff = arith::SubIOp::create(rewriter, loc, ji64, ii64);
            Value rowStartI64 = arith::AddIOp::create(rewriter, loc, diff, constLi64);
            Value diaRowStart =
                arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(), rowStartI64);

            // ---- rank-reducing extract_slice ---------------------------------
            // offsets = [diaRowStart, i]
            // sizes   = [sliceLen,    1]  <- dim-1 size is STATIC 1
            // strides = [1,           1]
            SmallVector<OpFoldResult> extractOffsets = { getAsOpFoldResult(diaRowStart),
                                                         getAsOpFoldResult(i) };
            SmallVector<OpFoldResult> extractSizes = { getAsOpFoldResult(sliceLen), staticOne };
            SmallVector<OpFoldResult> extractStrides = { staticStride, staticStride };

            Value slice = tensor::ExtractSliceOp::create(
                rewriter, loc, sliceType, diaIn, extractOffsets, extractSizes, extractStrides);

            // ==================================================================
            // Step 1 – row max (reduce over the band slice only)
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
            Value expEmpty =
                tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{ ShapedType::kDynamic },
                                        f32, ValueRange{ sliceLen });

            Value exps = linalg::GenericOp::create(
                             rewriter, loc, TypeRange{ expEmpty.getType() },
                             ValueRange{ slice, rowMaxTensor }, ValueRange{ expEmpty },
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
            Value normEmpty =
                tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{ ShapedType::kDynamic },
                                        f32, ValueRange{ sliceLen });

            Value normalized = linalg::GenericOp::create(
                                   rewriter, loc, TypeRange{ normEmpty.getType() },
                                   ValueRange{ exps, rowSumTensor }, ValueRange{ normEmpty },
                                   ArrayRef<AffineMap>{ map1D, mapSc, map1D }, par1D,
                                   [&](OpBuilder& b, Location loc, ValueRange args) {
                                       Value div = arith::DivFOp::create(b, loc, args[0], args[1]);
                                       linalg::YieldOp::create(b, loc, div);
                                   })
                                   .getResult(0);

            // ==================================================================
            // Step 5 – rank-expanding insert_slice back into the DIA tensor
            // ==================================================================
            SmallVector<OpFoldResult> insertOffsets = { getAsOpFoldResult(diaRowStart),
                                                        getAsOpFoldResult(i) };
            SmallVector<OpFoldResult> insertSizes = { getAsOpFoldResult(sliceLen), staticOne };
            SmallVector<OpFoldResult> insertStrides = { staticStride, staticStride };

            Value updatedDia = tensor::InsertSliceOp::create(
                rewriter, loc, normalized, diaIn, insertOffsets, insertSizes, insertStrides);

            scf::YieldOp::create(rewriter, loc, ValueRange{ updatedDia });
        }

        rowLoop->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, rowLoop.getResult(0));
        return success();
    }

    LogicalResult matchAndRewrite(dia::SoftmaxOp op, PatternRewriter& rewriter) const {
        auto dict = op->getAttrDictionary();
        if (!dict) dict = DictionaryAttr();

        const BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);
        if (!opBandInfo.IsDia) return denseBandedSoftmaxToLinalg(op, rewriter, opBandInfo);
        return diaBandedSoftmaxToLinalg(op, rewriter, opBandInfo);
    }
};

void addDIASoftmaxPatterns(RewritePatternSet& patterns) {
    patterns.add<DIASoftmaxPattern>(patterns.getContext());
}

}  // namespace mlir::bpa
//
