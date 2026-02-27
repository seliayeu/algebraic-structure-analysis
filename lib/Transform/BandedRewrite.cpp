#include "Transform/BandedRewrite.h"

#include <cstdint>

#include "Analysis/BandedStructureAnalysis.h"
#include "Utils/TransformUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

#define GEN_PASS_DEF_BANDEDREWRITE
#include "lib/Transform/Passes.h.inc"

// ------------------------------------------------------------------------------------------------------------------------------
// Matmul
// ------------------------------------------------------------------------------------------------------------------------------

struct MatMulPattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter& rewriter) const override {
        auto dict = op->getAttrDictionary();
        if (!dict) dict = DictionaryAttr();

        const BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        auto lower = opBandInfo.Property.LowerBandwidth;
        auto upper = opBandInfo.Property.UpperBandwidth;
        auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
        const uint64_t N = resultType.getDimSize(1);

        // TODO: check when to change the layout
        if (lower == 0 && upper == 0)
            return diagDenseRewriteToDIA(op, rewriter);
        else {
            if (shouldCompress(opBandInfo, N))
                return denseBandedRewriteToDIA(op, rewriter, opBandInfo);
            else
                return denseBandedRewrite(op, rewriter);
        }
        return failure();
    }

    /// Rewrite a dense `linalg.matmul` into an explicit SCF loop nest that
    /// computes only the entries within the intersection.
    ///
    /// The result is materialized in the DIA format.
    LogicalResult denseBandedRewriteToDIA(linalg::MatmulOp op, PatternRewriter& rewriter,
                                          const BandedSubMatrix& outputBand) const {
        Location loc = op.getLoc();
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutputs()[0];

        Operation* defOpA = A.getDefiningOp();
        Operation* defOpB = B.getDefiningOp();

        auto dictA = defOpA->getAttrDictionary();
        auto dictB = defOpB->getAttrDictionary();

        if (!dictA || !dictB) return failure();

        const BandedSubMatrix bandA = BandedStructureAnalysis::readPropertyFromDictAttr(dictA);
        const BandedSubMatrix bandB = BandedStructureAnalysis::readPropertyFromDictAttr(dictB);

        auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
        const uint64_t N = resultType.getDimSize(0);
        const uint64_t M = resultType.getDimSize(1);

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value dimN = arith::ConstantIndexOp::create(rewriter, loc, N);
        Value dimM = arith::ConstantIndexOp::create(rewriter, loc, M);

        Value lowerA = arith::ConstantIndexOp::create(rewriter, loc, bandA.Property.LowerBandwidth);
        Value upperA = arith::ConstantIndexOp::create(rewriter, loc, bandA.Property.UpperBandwidth);
        Value lowerB = arith::ConstantIndexOp::create(rewriter, loc, bandB.Property.LowerBandwidth);
        Value upperB = arith::ConstantIndexOp::create(rewriter, loc, bandB.Property.UpperBandwidth);
        Value lowerC =
            arith::ConstantIndexOp::create(rewriter, loc, outputBand.Property.LowerBandwidth);

        // DIA output type: (lC + uC + 1) x N
        const int64_t lC = outputBand.Property.LowerBandwidth;
        const int64_t uC = outputBand.Property.UpperBandwidth;
        int64_t numDiags = lC + uC + 1;
        auto elementType = resultType.getElementType();
        auto diaType = RankedTensorType::get({ numDiags, (int64_t)N }, elementType);

        Value emptyDia = tensor::EmptyOp::create(rewriter, loc, diaType, ValueRange{});
        Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                               rewriter.getZeroAttr(elementType));
        Value zeroedDia =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ emptyDia })
                .getResult(0);

        Value lCVal = arith::ConstantIndexOp::create(rewriter, loc, lC);
        Value uCVal = arith::ConstantIndexOp::create(rewriter, loc, uC);

        auto iLoop = scf::ForOp::create(
            rewriter, loc, c0, dimN, c1, ValueRange{ zeroedDia },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange outerArgs) {
                Value cOuter = outerArgs[0];

                // j start = max(0, i - (La + Lb))
                Value lAPlusLb = arith::AddIOp::create(ob, loc, lowerA, lowerB);
                Value iMinusLower = arith::SubIOp::create(ob, loc, i, lAPlusLb);
                Value jStart = arith::MaxSIOp::create(ob, loc, c0, iMinusLower);

                // j end = min(M, i + (Ua + Ub))
                Value uAPlusUb = arith::AddIOp::create(ob, loc, upperA, upperB);
                Value iPlusUpper = arith::AddIOp::create(ob, loc, i, uAPlusUb);
                Value mMinusOne = arith::AddIOp::create(ob, loc, dimM, c1);
                Value iPlusUpperP1 = arith::AddIOp::create(ob, loc, iPlusUpper, c1);
                Value jEnd = arith::MinSIOp::create(ob, loc, dimM, iPlusUpperP1);

                auto jLoop = scf::ForOp::create(
                    ob, loc, jStart, jEnd, c1, ValueRange{ cOuter },
                    [&](OpBuilder& mb, Location loc, Value j, ValueRange midArgs) {
                        Value cMid = midArgs[0];

                        Value initAcc = arith::ConstantOp::create(
                            mb, loc, elementType, rewriter.getZeroAttr(elementType));

                        // k start = max(i-La, j - Ub)
                        Value iMinusLa = arith::SubIOp::create(mb, loc, i, lowerA);
                        Value jMinusUb = arith::SubIOp::create(mb, loc, j, upperB);
                        Value kStart = arith::MaxSIOp::create(mb, loc, jMinusUb, iMinusLa);

                        // k end = min(i + Ua, j + Lb)
                        Value iPlusUa = arith::AddIOp::create(mb, loc, i, upperA);
                        Value jPlusLb = arith::AddIOp::create(mb, loc, j, lowerB);
                        Value kEndPlusOne = arith::AddIOp::create(
                            mb, loc, arith::MinSIOp::create(mb, loc, jPlusLb, iPlusUa), c1);
                        Value kEnd = arith::MinSIOp::create(
                            mb, loc, kEndPlusOne, arith::ConstantIndexOp::create(mb, loc, N));

                        auto kLoop = scf::ForOp::create(
                            mb, loc, kStart, kEnd, c1, ValueRange{ cMid, initAcc },
                            [&](OpBuilder& ib, Location loc, Value k, ValueRange innerArgs) {
                                Value cInner = innerArgs[0];
                                Value acc = innerArgs[1];

                                Value aVal =
                                    tensor::ExtractOp::create(ib, loc, A, ValueRange{ i, k });
                                Value bVal =
                                    tensor::ExtractOp::create(ib, loc, B, ValueRange{ k, j });
                                Value mul = arith::MulFOp::create(ib, loc, aVal, bVal);
                                Value add = arith::AddFOp::create(ib, loc, acc, mul);

                                scf::YieldOp::create(ib, loc, ValueRange{ cInner, add });
                            });

                        // Write to DIA: data[j - i + lC, i]
                        Value diagIdx = arith::AddIOp::create(
                            mb, loc, arith::SubIOp::create(mb, loc, j, i), lCVal);
                        Value updated =
                            tensor::InsertOp::create(mb, loc, kLoop.getResult(1),
                                                     kLoop.getResult(0), ValueRange{ diagIdx, i });

                        scf::YieldOp::create(mb, loc, ValueRange{ updated });
                    });

                scf::YieldOp::create(ob, loc, ValueRange{ jLoop.getResult(0) });
            });
        iLoop->setAttr("metadata", getMetadataWithTensorLayout(*op, rewriter, "dia"));
        rewriter.replaceOp(op, iLoop.getResult(0));
        return success();
    }

    LogicalResult diagDenseRewriteToDIA(linalg::MatmulOp op, PatternRewriter& rewriter) const {
        auto loc = op.getLoc();
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutputs()[0];
        MLIRContext* context = rewriter.getContext();

        auto matrixType = cast<RankedTensorType>(C.getType());
        int64_t M = matrixType.getDimSize(0);
        auto elementType = matrixType.getElementType();

        // tensor<M x f64>
        auto vectorType = RankedTensorType::get({ M }, elementType);
        Value emptyVec = tensor::EmptyOp::create(rewriter, loc, vectorType, ValueRange{});
        C.replaceAllUsesWith(emptyVec);

        AffineExpr d0 = rewriter.getAffineDimExpr(0);
        AffineMap diagMap = AffineMap::get(1, 0, { d0, d0 }, context);

        AffineMap vectorMap = AffineMap::get(1, 0, { d0 }, context);

        SmallVector<AffineMap, 3> indexingMaps = {
            diagMap,
            diagMap,
            vectorMap,
        };

        SmallVector<utils::IteratorType, 1> iteratorTypes = { utils::IteratorType::parallel };

        auto genericOp = linalg::GenericOp::create(
            rewriter, loc, TypeRange{ vectorType }, ValueRange{ A, B }, ValueRange{ emptyVec },
            indexingMaps, iteratorTypes, [&](OpBuilder& b, Location loc, ValueRange args) {
                Value mul = arith::MulFOp::create(b, loc, args[0], args[1]);
                linalg::YieldOp::create(b, loc, ValueRange{ mul });
            });

        auto layout = std::string("dia");
        genericOp->setAttr("metadata", getMetadataWithTensorLayout(*op, rewriter, layout));
        rewriter.replaceOp(op, genericOp);

        return success();
    }

    LogicalResult diagRewrite(linalg::MatmulOp op, PatternRewriter& rewriter) const {
        auto loc = op.getLoc();
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutputs()[0];
        MLIRContext* context = rewriter.getContext();

        AffineExpr d0 = rewriter.getAffineDimExpr(0);

        AffineMap diagMap = AffineMap::get(1, 0, { d0, d0 }, context);

        SmallVector<AffineMap, 3> indexingMaps = {
            diagMap,
            diagMap,
            diagMap,
        };

        llvm::SmallVector<utils::IteratorType, 1> iteratorTypes = { utils::IteratorType::parallel };

        auto genericOp = linalg::GenericOp::create(
            rewriter, loc, TypeRange{ op.getResult(0).getType() }, ValueRange{ A, B },
            ValueRange{ C }, indexingMaps, iteratorTypes,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                Value mul = arith::MulFOp::create(b, loc, args[0], args[1]);
                linalg::YieldOp::create(b, loc, ValueRange{ mul });
            });
        genericOp->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, genericOp);
        return success();
    }

    LogicalResult denseBandedRewrite(linalg::MatmulOp op, PatternRewriter& rewriter) const {
        Location loc = op.getLoc();
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutputs()[0];

        Operation* defOpA = A.getDefiningOp();
        Operation* defOpB = B.getDefiningOp();

        auto dictA = defOpA->getAttrDictionary();
        auto dictB = defOpB->getAttrDictionary();

        if (!dictA || !dictB) return failure();

        BandedSubMatrix bandA = BandedStructureAnalysis::readPropertyFromDictAttr(dictA);
        BandedSubMatrix bandB = BandedStructureAnalysis::readPropertyFromDictAttr(dictB);

        auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
        const uint64_t N = resultType.getDimSize(0);
        const uint64_t M = resultType.getDimSize(1);

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value dimN = arith::ConstantIndexOp::create(rewriter, loc, N);
        Value dimM = arith::ConstantIndexOp::create(rewriter, loc, M);

        Value lowerA = arith::ConstantIndexOp::create(rewriter, loc, bandA.Property.LowerBandwidth);
        Value upperA = arith::ConstantIndexOp::create(rewriter, loc, bandA.Property.UpperBandwidth);
        Value lowerB = arith::ConstantIndexOp::create(rewriter, loc, bandB.Property.LowerBandwidth);
        Value upperB = arith::ConstantIndexOp::create(rewriter, loc, bandB.Property.UpperBandwidth);

        // C is the projection of the Minkowski sum of A and B bands.
        //----------------------------------------------------------
        // for i in [0, N)
        //  for k in  [max(0, i-lower), min(M, i+upper)]
        //    for j in [max(i - La, k - Ub), min(i + Ua, k + Lb)]
        //        C[i,k] += A[i,j] * B[j,k]
        auto iLoop = scf::ForOp::create(
            rewriter, loc, c0, dimN, c1, ValueRange{ C },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                // C
                Value cOut = iArgs[0];
                // k start = max(0, i - (La + Lb))
                Value lAPlusLb = arith::AddIOp::create(ob, loc, lowerA, lowerB);
                Value iMinusLower = arith::SubIOp::create(ob, loc, i, lAPlusLb);
                Value kStart = arith::MaxSIOp::create(ob, loc, c0, iMinusLower);

                // k end = min(M, i + (Ua + Ub))
                Value uAPlusUb = arith::AddIOp::create(ob, loc, upperA, upperB);
                Value iPlusUpper = arith::AddIOp::create(ob, loc, i, uAPlusUb);
                Value mMinusOne = arith::AddIOp::create(ob, loc, dimM, c1);
                Value iPlusUpperP1 = arith::AddIOp::create(ob, loc, iPlusUpper, c1);
                Value kEnd = arith::MinSIOp::create(ob, loc, dimM, iPlusUpperP1);
                // k loop
                auto kLoop = scf::ForOp::create(
                    ob, loc, kStart, kEnd, c1, ValueRange{ cOut },
                    [&](OpBuilder& mb, Location loc, Value k, ValueRange kArgs) {
                        auto cMid = kArgs[0];
                        // j start = max(i-La, k - Ub)
                        Value iMinusLa = arith::SubIOp::create(mb, loc, i, lowerA);
                        Value kMinusUb = arith::SubIOp::create(mb, loc, k, upperB);
                        Value jStart = arith::MaxSIOp::create(mb, loc, kMinusUb, iMinusLa);

                        // j end = min(i + Ua, k + Lb)
                        Value iPlusUa = arith::AddIOp::create(mb, loc, i, upperA);
                        Value kPlusLb = arith::AddIOp::create(mb, loc, k, lowerB);
                        Value jEndPlusOne = arith::AddIOp::create(
                            mb, loc, arith::MinSIOp::create(mb, loc, kPlusLb, iPlusUa), c1);
                        Value jEnd = arith::MinSIOp::create(
                            mb, loc, jEndPlusOne, arith::ConstantIndexOp::create(mb, loc, N));
                        auto jLoop = scf::ForOp::create(
                            mb, loc, jStart, jEnd, c1, ValueRange{ cMid },
                            [&](OpBuilder& ib, Location loc, Value j, ValueRange jArgs) {
                                Value cInner = jArgs[0];
                                // Load current C[i,k]
                                Value cik =
                                    tensor::ExtractOp::create(ib, loc, cInner, ValueRange{ i, k });
                                // Load A[i,j] and B[j,k]
                                Value aij =
                                    tensor::ExtractOp::create(ib, loc, A, ValueRange{ i, j });
                                Value bjk =
                                    tensor::ExtractOp::create(ib, loc, B, ValueRange{ j, k });
                                // C[i,k] += A[i,j] * B[j,k]
                                Value mul = arith::MulFOp::create(ib, loc, aij, bjk);
                                Value add = arith::AddFOp::create(ib, loc, cik, mul);
                                // Insert updated value back
                                Value updated = tensor::InsertOp::create(ib, loc, add, cInner,
                                                                         ValueRange{ i, k });
                                scf::YieldOp::create(ib, loc, ValueRange{ updated });
                            });
                        scf::YieldOp::create(mb, loc, jLoop.getResults());
                    });
                scf::YieldOp::create(ob, loc, kLoop.getResults());
            });
        iLoop->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, iLoop.getResult(0));
        return success();
    }
};

// ------------------------------------------------------------------------------------------------------------------------------
// ElementwiseOp
// ------------------------------------------------------------------------------------------------------------------------------

struct GenericElementWisePattern : public OpRewritePattern<linalg::ElementwiseOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::ElementwiseOp op,
                                  PatternRewriter& rewriter) const override {
        if (op.getKind() != linalg::ElementwiseKind::mul &&
            op.getKind() != linalg::ElementwiseKind::add &&
            op.getKind() != linalg::ElementwiseKind::sub)
            return failure();

        auto dict = op->getAttrDictionary();

        if (!dict) dict = DictionaryAttr();

        BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        auto lower = opBandInfo.Property.LowerBandwidth;
        auto upper = opBandInfo.Property.UpperBandwidth;
        auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
        uint64_t n = resultType.getDimSize(0);
        if (lower == 0 && upper == 0)
            return diagRewrite(op, rewriter);
        else
            return bandedRewrite(op, rewriter);
    }

    LogicalResult diagRewrite(linalg::ElementwiseOp op, PatternRewriter& rewriter) const {
        auto loc = op.getLoc();
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutputs()[0];
        MLIRContext* context = rewriter.getContext();

        AffineExpr d0 = rewriter.getAffineDimExpr(0);

        AffineMap diagMap = AffineMap::get(1, 0, { d0, d0 }, context);

        SmallVector<AffineMap, 3> indexingMaps = {
            diagMap,
            diagMap,
            diagMap,
        };

        llvm::SmallVector<utils::IteratorType, 1> iteratorTypes = { utils::IteratorType::parallel };

        auto genericOp = linalg::GenericOp::create(
            rewriter, loc, TypeRange{ op.getResult(0).getType() }, ValueRange{ A, B },
            ValueRange{ C }, indexingMaps, iteratorTypes,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                auto lhs = args[0];
                auto rhs = args[1];
                Value opResult = getInnerArithOp(op, b, loc, lhs, rhs);
                linalg::YieldOp::create(b, loc, ValueRange{ opResult });
            });
        genericOp->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, genericOp);
        return success();
    }

    LogicalResult bandedRewrite(linalg::ElementwiseOp op, PatternRewriter& rewriter) const {
        Location loc = op.getLoc();
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutputs()[0];

        Operation* defOpA = A.getDefiningOp();
        Operation* defOpB = B.getDefiningOp();

        auto dictA = defOpA->getAttrDictionary();
        auto dictB = defOpB->getAttrDictionary();

        if (!dictA || !dictB) return failure();

        BandedSubMatrix bandA = BandedStructureAnalysis::readPropertyFromDictAttr(dictA);
        BandedSubMatrix bandB = BandedStructureAnalysis::readPropertyFromDictAttr(dictB);

        auto lowerA = bandA.Property.LowerBandwidth;
        auto upperA = bandA.Property.UpperBandwidth;
        auto upperB = bandB.Property.UpperBandwidth;
        auto lowerB = bandB.Property.LowerBandwidth;

        // takes only the intersection
        uint64_t lower = std::min(bandA.Property.LowerBandwidth, bandB.Property.LowerBandwidth);
        uint64_t upper = std::min(bandB.Property.UpperBandwidth, bandB.Property.UpperBandwidth);

        auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
        const int64_t N = resultType.getDimSize(0);
        const int64_t M = resultType.getDimSize(1);

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value dimN = arith::ConstantIndexOp::create(rewriter, loc, N);
        Value lowerBW = arith::ConstantIndexOp::create(rewriter, loc, (int64_t)lower);
        Value upperBW = arith::ConstantIndexOp::create(rewriter, loc, (int64_t)upper);

        auto iLoop = scf::ForOp::create(
            rewriter, loc, c0, dimN, c1, ValueRange{ C },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                Value cOut = iArgs[0];

                // j start = max(0, i - lower)
                Value iMinusLower = arith::SubIOp::create(ob, loc, i, lowerBW);
                Value jStart = arith::MaxSIOp::create(ob, loc, c0, iMinusLower);

                // j end = min(M, i + upper + 1)
                Value iPlusUpper = arith::AddIOp::create(ob, loc, i, upperBW);
                Value iPlusUpperP1 = arith::AddIOp::create(ob, loc, iPlusUpper, c1);
                Value jEnd = arith::MinSIOp::create(
                    ob, loc, arith::ConstantIndexOp::create(ob, loc, M), iPlusUpperP1);

                auto jLoop = scf::ForOp::create(
                    ob, loc, jStart, jEnd, c1, ValueRange{ cOut },
                    [&](OpBuilder& ib, Location loc, Value j, ValueRange jArgs) {
                        Value cInner = jArgs[0];

                        Value aij = tensor::ExtractOp::create(ib, loc, A, ValueRange{ i, j });
                        Value bij = tensor::ExtractOp::create(ib, loc, B, ValueRange{ i, j });
                        Value mul = arith::MulFOp::create(ib, loc, aij, bij);
                        Value opResult = getInnerArithOp(op, ib, loc, aij, bij);
                        Value updated =
                            tensor::InsertOp::create(ib, loc, opResult, cInner, ValueRange{ i, j });
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    });

                scf::YieldOp::create(ob, loc, jLoop.getResults());
            });

        iLoop->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, iLoop.getResult(0));
        return success();
    }

   private:
    Value getInnerArithOp(linalg::ElementwiseOp op, OpBuilder& ob, Location& loc, Value& lhs,
                          Value& rhs) const {
        auto kind = op.getKind();
        Value result;
        if (kind == linalg::ElementwiseKind::mul)
            result = arith::MulFOp::create(ob, loc, lhs, rhs);
        else if (kind == linalg::ElementwiseKind::add)
            result = arith::AddFOp::create(ob, loc, lhs, rhs);
        else
            result = arith::SubFOp::create(ob, loc, lhs, rhs);
        return result;
    }
};

// ------------------------------------------------------------------------------------------------------------------------------
// Transposition
// ------------------------------------------------------------------------------------------------------------------------------

struct TransposePattern : public OpRewritePattern<linalg::TransposeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                  PatternRewriter& rewriter) const override {
        auto dict = op->getAttrDictionary();
        if (!dict) dict = DictionaryAttr();
        BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        if (opBandInfo.Property.LowerBandwidth == 0 && opBandInfo.Property.UpperBandwidth == 0) {
            rewriter.replaceOp(op, op.getInput());
            return success();
        }

        auto inputUpper = opBandInfo.Property.LowerBandwidth;
        auto inputLower = opBandInfo.Property.UpperBandwidth;

        MLIRContext* context = rewriter.getContext();

        AffineExpr d0 = rewriter.getAffineDimExpr(0);
        AffineExpr d1 = rewriter.getAffineDimExpr(1);

        AffineMap inputMap = AffineMap::get(2, 0, { d0, d1 }, context);
        AffineMap outputMap = AffineMap::get(2, 0, { d1, d0 }, context);

        SmallVector<AffineMap, 2> indexingMaps = { inputMap, outputMap };

        llvm::SmallVector<utils::IteratorType, 1> iteratorTypes = { utils::IteratorType::parallel,
                                                                    utils::IteratorType::parallel };

        Location loc = op->getLoc();
        auto input = op->getOperand(0);
        auto output = op->getOperand(1);

        auto genericOp = linalg::GenericOp::create(
            rewriter, loc, TypeRange{ op.getResult().getType() }, ValueRange{ input },
            ValueRange{ output }, indexingMaps, iteratorTypes,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                linalg::YieldOp::create(b, loc, ValueRange{ args[0] });
            });
        genericOp->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, genericOp);
        return success();
    }
};

struct BandedRewrite : public impl::BandedRewriteBase<BandedRewrite> {
    using BandedRewriteBase::BandedRewriteBase;

    void runOnOperation() override {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = funcOp.getContext();

        RewritePatternSet patterns(context);

        patterns.add<MatMulPattern, GenericElementWisePattern, TransposePattern>(context);

        GreedyRewriteConfig config;
        config.setMaxIterations(1);
        config.setUseTopDownTraversal(true);

        (void)applyPatternsGreedily(funcOp, std::move(patterns), config);

        funcOp.walk([&](func::ReturnOp returnOp) {
            SmallVector<Type> newTypes;
            bool changed = false;

            for (Value operand : returnOp.getOperands()) {
                newTypes.push_back(operand.getType());
                Operation* defOp = operand.getDefiningOp();
                if (!defOp) continue;
                auto metadata = defOp->getAttrOfType<DictionaryAttr>("metadata");
                if (!metadata) continue;
                auto layout = metadata.getAs<StringAttr>("layout");
                // WARNING: This only works for squared matrices right now!
                if (layout && layout.getValue() == "dia") changed = true;
            }

            if (!changed) return;

            auto newFuncType =
                FunctionType::get(context, funcOp.getFunctionType().getInputs(), newTypes);
            funcOp.setType(newFuncType);
        });
    }
    // TODO: you don't need an extra metadata flag for layout update.
    // Create a helper function to match bands with tensor layouts.
    // This will assume that all supported ops lower to the correct tensor layout.
};
}  // namespace mlir::bpa
