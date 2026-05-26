#include "Transform/BandedRewrite.h"

#include <cstdint>
#include <optional>

#include "Analysis/BandedStructureAnalysis.h"
#include "Transform/BandedPropagation.h"
#include "Transform/Operators/DIABatchMatmulRewrite.h"
#include "Transform/Operators/DIAElementwiseRewrite.h"
#include "Transform/Operators/DIAMatmulRewrite.h"
#include "Transform/Operators/DIASoftmaxRewrite.h"
#include "Transform/Operators/DIATransposeRewrite.h"
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

struct BatchMatmulPattern : public OpRewritePattern<linalg::BatchMatmulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult denseTimesDenseToDenseDiagBatchMatmulToLinalg(linalg::BatchMatmulOp op,
                                                                PatternRewriter& rewriter) const {
        auto loc = op.getLoc();
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutputs()[0];
        MLIRContext* context = rewriter.getContext();

        AffineExpr d0 = rewriter.getAffineDimExpr(0);
        AffineExpr d1 = rewriter.getAffineDimExpr(1);

        AffineMap batchDiagMap = AffineMap::get(2, 0, { d0, d1, d1 }, context);

        SmallVector<AffineMap, 3> indexingMaps = { batchDiagMap, batchDiagMap, batchDiagMap };
        SmallVector<utils::IteratorType, 2> iteratorTypes = { utils::IteratorType::parallel,
                                                              utils::IteratorType::parallel };

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

    LogicalResult denseTimesDenseToDenseBandedBatchMatmulToSCF(linalg::BatchMatmulOp op,
                                                               PatternRewriter& rewriter) const {
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
        const uint64_t K = resultType.getDimSize(0);
        const uint64_t N = resultType.getDimSize(1);
        const uint64_t M = resultType.getDimSize(2);

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value dimK = arith::ConstantIndexOp::create(rewriter, loc, K);
        Value dimN = arith::ConstantIndexOp::create(rewriter, loc, N);
        Value dimM = arith::ConstantIndexOp::create(rewriter, loc, M);

        Value lowerA = arith::ConstantIndexOp::create(rewriter, loc, bandA.Property.LowerBandwidth);
        Value upperA = arith::ConstantIndexOp::create(rewriter, loc, bandA.Property.UpperBandwidth);
        Value lowerB = arith::ConstantIndexOp::create(rewriter, loc, bandB.Property.LowerBandwidth);
        Value upperB = arith::ConstantIndexOp::create(rewriter, loc, bandB.Property.UpperBandwidth);

        // Same as before but with an extra loop around the batch
        // for b in [0, K):
        //   for i in [0, N):
        //     for k in [max(0, i-(lA+lB)), min(M, i+(uA+uB)+1)):
        //       for j in [max(0, max(i-lA, k-uB)), min(N, min(i+uA, k+lB)+1)):
        //         C[b,i,k] += A[b,i,j] * B[b,j,k]

        auto bLoop = scf::ForOp::create(
            rewriter, loc, c0, dimK, c1, ValueRange{ C },
            [&](OpBuilder& bb, Location loc, Value b, ValueRange bArgs) {
                Value cBatch = bArgs[0];
                auto iLoop = scf::ForOp::create(
                    rewriter, loc, c0, dimN, c1, ValueRange{ cBatch },
                    [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                        Value cOut = iArgs[0];

                        Value iMinusLa = arith::SubIOp::create(ob, loc, i, lowerA);
                        Value jStart = arith::MaxSIOp::create(ob, loc, c0, iMinusLa);

                        Value iPlusUa = arith::AddIOp::create(ob, loc, i, upperA);
                        Value iPlusUaP1 = arith::AddIOp::create(ob, loc, iPlusUa, c1);
                        Value jEnd = arith::MinSIOp::create(ob, loc, dimN, iPlusUaP1);

                        auto jLoop = scf::ForOp::create(
                            ob, loc, jStart, jEnd, c1, ValueRange{ cOut },
                            [&](OpBuilder& mb, Location loc, Value j, ValueRange jArgs) {
                                Value cMid = jArgs[0];

                                Value aij =
                                    tensor::ExtractOp::create(mb, loc, A, ValueRange{ b, i, j });

                                Value jMinusLb = arith::SubIOp::create(mb, loc, j, lowerB);
                                Value kStart = arith::MaxSIOp::create(mb, loc, c0, jMinusLb);

                                Value jPlusUb = arith::AddIOp::create(mb, loc, j, upperB);
                                Value jPlusUbP1 = arith::AddIOp::create(mb, loc, jPlusUb, c1);
                                Value kEnd = arith::MinSIOp::create(mb, loc, dimM, jPlusUbP1);

                                auto kLoop = scf::ForOp::create(
                                    mb, loc, kStart, kEnd, c1, ValueRange{ cMid },
                                    [&](OpBuilder& ib, Location loc, Value k, ValueRange kArgs) {
                                        Value cInner = kArgs[0];
                                        Value cik = tensor::ExtractOp::create(
                                            ib, loc, cInner, ValueRange{ b, i, k });
                                        Value bjk = tensor::ExtractOp::create(
                                            ib, loc, B, ValueRange{ b, j, k });
                                        Value mul = arith::MulFOp::create(ib, loc, aij, bjk);
                                        Value add = arith::AddFOp::create(ib, loc, cik, mul);
                                        Value updated = tensor::InsertOp::create(
                                            ib, loc, add, cInner, ValueRange{ b, i, k });
                                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                                    });
                                scf::YieldOp::create(mb, loc, kLoop.getResults());
                            });
                        scf::YieldOp::create(ob, loc, jLoop.getResults());
                    });
                iLoop->setAttr("metadata", op->getAttr("metadata"));

                scf::YieldOp::create(bb, loc, iLoop.getResults());
            });

        bLoop->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, bLoop.getResult(0));
        return success();
    }

    LogicalResult matchAndRewrite(linalg::BatchMatmulOp op,
                                  PatternRewriter& rewriter) const override {
        auto dict = op->getAttrDictionary();
        if (!dict) dict = DictionaryAttr();

        const BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        if (opBandInfo.isDiagonal())
            return denseTimesDenseToDenseDiagBatchMatmulToLinalg(op, rewriter);
        else
            return denseTimesDenseToDenseBandedBatchMatmulToSCF(op, rewriter);
        return failure();
    }
};

// ------------------------------------------------------------------------------------------------------------------------------
// linalg.MatmulOp
// ------------------------------------------------------------------------------------------------------------------------------

struct MatMulPattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter& rewriter) const override {
        auto dict = op->getAttrDictionary();
        if (!dict) dict = DictionaryAttr();

        const BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Operation* defOpA = A.getDefiningOp();
        Operation* defOpB = B.getDefiningOp();

        auto dictA = defOpA->getAttrDictionary();
        auto dictB = defOpB->getAttrDictionary();

        if (!dictA || !dictB) return failure();

        const BandedSubMatrix bandA = BandedStructureAnalysis::readPropertyFromDictAttr(dictA);
        const BandedSubMatrix bandB = BandedStructureAnalysis::readPropertyFromDictAttr(dictB);

        auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
        const uint64_t MAX = resultType.getDimSize(1) - 1;

        // if (isFullyDense(bandA, bandB, opBandInfo, MAX)) return failure();

        if (bandA.isDiagonal() && bandB.isDiagonal() && opBandInfo.isDiagonal())
            return denseTimesDenseToDenseDiagMatmulToLinalg(op, rewriter);
        // banded
        else
            return denseTimesDenseToDenseBandedMatmulToSCF(op, rewriter, bandA, bandB, opBandInfo);
        return failure();
    }

    LogicalResult denseTimesDenseToDenseDiagMatmulToLinalg(linalg::MatmulOp op,
                                                           PatternRewriter& rewriter) const {
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

    LogicalResult denseTimesDenseToDenseBandedMatmulToSCF(linalg::MatmulOp op,
                                                          PatternRewriter& rewriter,
                                                          const BandedSubMatrix& bandA,
                                                          const BandedSubMatrix& bandB,
                                                          const BandedSubMatrix& resultBand) const {
        Location loc = op.getLoc();
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutputs()[0];

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
            arith::ConstantIndexOp::create(rewriter, loc, resultBand.Property.LowerBandwidth);
        Value upperC =
            arith::ConstantIndexOp::create(rewriter, loc, resultBand.Property.UpperBandwidth);

        auto elementType = resultType.getElementType();
        Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                               rewriter.getZeroAttr(elementType));
        Value zeroedC =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0);

        auto iLoop = scf::ForOp::create(
            rewriter, loc, c0, dimN, c1, ValueRange{ zeroedC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                Value cOut = iArgs[0];

                Value iMinusLa = arith::SubIOp::create(ob, loc, i, lowerA);
                Value jStart = arith::MaxSIOp::create(ob, loc, c0, iMinusLa);

                Value iPlusUa = arith::AddIOp::create(ob, loc, i, upperA);
                Value iPlusUaP1 = arith::AddIOp::create(ob, loc, iPlusUa, c1);
                Value jEnd = arith::MinSIOp::create(ob, loc, dimN, iPlusUaP1);

                auto jLoop = scf::ForOp::create(
                    ob, loc, jStart, jEnd, c1, ValueRange{ cOut },
                    [&](OpBuilder& mb, Location loc, Value j, ValueRange jArgs) {
                        Value cMid = jArgs[0];

                        Value aij = tensor::ExtractOp::create(mb, loc, A, ValueRange{ i, j });

                        Value jMinusLb = arith::SubIOp::create(mb, loc, j, lowerB);
                        Value iMinusLc = arith::SubIOp::create(mb, loc, i, lowerC);

                        Value maxJB_IC = arith::MaxSIOp::create(mb, loc, jMinusLb, iMinusLc);
                        Value kStart = arith::MaxSIOp::create(mb, loc, c0, maxJB_IC);

                        Value jPlusUb = arith::AddIOp::create(mb, loc, j, upperB);
                        Value iPlusUc = arith::AddIOp::create(mb, loc, i, upperC);

                        Value jPlusUbP1 = arith::AddIOp::create(mb, loc, jPlusUb, c1);
                        Value iPlusUcP1 = arith::AddIOp::create(mb, loc, iPlusUc, c1);

                        Value minJB_IC = arith::MinSIOp::create(mb, loc, jPlusUbP1, iPlusUcP1);
                        Value kEnd = arith::MinSIOp::create(mb, loc, dimM, minJB_IC);

                        auto kLoop = scf::ForOp::create(
                            mb, loc, kStart, kEnd, c1, ValueRange{ cMid },
                            [&](OpBuilder& ib, Location loc, Value k, ValueRange kArgs) {
                                Value cInner = kArgs[0];
                                Value cik =
                                    tensor::ExtractOp::create(ib, loc, cInner, ValueRange{ i, k });
                                Value bjk =
                                    tensor::ExtractOp::create(ib, loc, B, ValueRange{ j, k });
                                Value mul = arith::MulFOp::create(ib, loc, aij, bjk);
                                Value add = arith::AddFOp::create(ib, loc, cik, mul);
                                Value updated = tensor::InsertOp::create(ib, loc, add, cInner,
                                                                         ValueRange{ i, k });
                                scf::YieldOp::create(ib, loc, ValueRange{ updated });
                            });
                        scf::YieldOp::create(mb, loc, kLoop.getResults());
                    });
                scf::YieldOp::create(ob, loc, jLoop.getResults());
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
        if (opBandInfo.isDiagonal())
            return denseTimesDenseToDenseDiagElementwiseToLinalg(op, rewriter);
        else
            return denseTimesDenseToDenseBandedElementwiseToSCF(op, rewriter);
    }

    LogicalResult denseTimesDenseToDenseDiagElementwiseToLinalg(linalg::ElementwiseOp op,
                                                                PatternRewriter& rewriter) const {
        auto loc = op.getLoc();
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutputs()[0];
        MLIRContext* context = rewriter.getContext();

        auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
        int64_t rank = resultType.getRank();

        SmallVector<AffineExpr> exprs;
        SmallVector<utils::IteratorType> iteratorTypes;

        if (rank == 2) {
            AffineExpr d0 = rewriter.getAffineDimExpr(0);
            exprs = { d0, d0 };
            iteratorTypes = { utils::IteratorType::parallel };
        } else if (rank == 3) {
            AffineExpr d0 = rewriter.getAffineDimExpr(0);
            AffineExpr d1 = rewriter.getAffineDimExpr(1);
            exprs = { d0, d1, d1 };
            iteratorTypes = { utils::IteratorType::parallel, utils::IteratorType::parallel };
        } else {
            return failure();
        }

        Type elementType = resultType.getElementType();
        auto zeroAttr = cast<TypedAttr>(rewriter.getZeroAttr(elementType));
        Value zeroConst = arith::ConstantOp::create(rewriter, loc, elementType, zeroAttr);
        auto fillOp =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zeroConst }, ValueRange{ C });

        Value filledC = fillOp.getResult(0);
        AffineMap diagMap = AffineMap::get(iteratorTypes.size(), 0, exprs, context);
        SmallVector<AffineMap, 3> indexingMaps = { diagMap, diagMap, diagMap };

        auto genericOp = linalg::GenericOp::create(
            rewriter, loc, TypeRange{ op.getResult(0).getType() }, ValueRange{ A, B },
            ValueRange{ filledC }, indexingMaps, iteratorTypes,
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

    LogicalResult denseTimesDenseToDenseBandedElementwiseToSCF(linalg::ElementwiseOp op,
                                                               PatternRewriter& rewriter) const {
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

        uint64_t lower = std::min(bandA.Property.LowerBandwidth, bandB.Property.LowerBandwidth);
        uint64_t upper = std::min(bandA.Property.UpperBandwidth, bandB.Property.UpperBandwidth);
        auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
        int64_t rank = resultType.getRank();

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Type elementType = resultType.getElementType();
        auto zeroAttr = cast<TypedAttr>(rewriter.getZeroAttr(elementType));
        Value zeroConst = arith::ConstantOp::create(rewriter, loc, elementType, zeroAttr);
        auto fillOp =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zeroConst }, ValueRange{ C });

        Value filledC = fillOp.getResult(0);

        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

        Value dimN = arith::ConstantIndexOp::create(rewriter, loc, resultType.getDimSize(rank - 2));
        Value dimM = arith::ConstantIndexOp::create(rewriter, loc, resultType.getDimSize(rank - 1));
        Value lowerBW = arith::ConstantIndexOp::create(rewriter, loc, (int64_t)lower);
        Value upperBW = arith::ConstantIndexOp::create(rewriter, loc, (int64_t)upper);

        auto buildInnerBandedLoops = [&](OpBuilder& builder, Location loc, Value cInOut,
                                         std::optional<Value> batchIndex) -> Value {
            auto iLoop = scf::ForOp::create(
                builder, loc, c0, dimN, c1, ValueRange{ cInOut },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    Value cOut = iArgs[0];
                    Value iMinusLower = arith::SubIOp::create(ob, loc, i, lowerBW);
                    Value jStart = arith::MaxSIOp::create(ob, loc, c0, iMinusLower);
                    Value iPlusUpper = arith::AddIOp::create(ob, loc, i, upperBW);
                    Value iPlusUpperP1 = arith::AddIOp::create(ob, loc, iPlusUpper, c1);
                    Value jEnd = arith::MinSIOp::create(ob, loc, dimM, iPlusUpperP1);
                    auto jLoop = scf::ForOp::create(
                        ob, loc, jStart, jEnd, c1, ValueRange{ cOut },
                        [&](OpBuilder& ib, Location loc, Value j, ValueRange jArgs) {
                            Value cInner = jArgs[0];

                            SmallVector<Value> indices;
                            if (batchIndex) indices.push_back(*batchIndex);
                            indices.push_back(i);
                            indices.push_back(j);

                            Value aij = tensor::ExtractOp::create(ib, loc, A, indices);
                            Value bij = tensor::ExtractOp::create(ib, loc, B, indices);
                            Value opResult = getInnerArithOp(op, ib, loc, aij, bij);
                            Value updated =
                                tensor::InsertOp::create(ib, loc, opResult, cInner, indices);

                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, jLoop.getResults());
                });
            return iLoop.getResult(0);
        };

        if (rank == 2) {
            Value result = buildInnerBandedLoops(rewriter, loc, filledC, std::nullopt);
            if (auto metadataAttr = op->getAttr("metadata")) {
                result.getDefiningOp()->setAttr("metadata", metadataAttr);
            }
            rewriter.replaceOp(op, result);
        } else if (rank == 3) {
            Value dimBatch =
                arith::ConstantIndexOp::create(rewriter, loc, resultType.getDimSize(0));
            auto bLoop = scf::ForOp::create(
                rewriter, loc, c0, dimBatch, c1, ValueRange{ filledC },
                [&](OpBuilder& batchBuilder, Location loc, Value b, ValueRange bArgs) {
                    Value result = buildInnerBandedLoops(batchBuilder, loc, bArgs[0], b);
                    scf::YieldOp::create(batchBuilder, loc, result);
                });
            bLoop->setAttr("metadata", op->getAttr("metadata"));
            rewriter.replaceOp(op, bLoop.getResult(0));
        } else {
            return failure();
        }
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
// linalg.tranpose
// ------------------------------------------------------------------------------------------------------------------------------

struct TransposePattern : public OpRewritePattern<linalg::TransposeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult denseBandedTranspose(linalg::TransposeOp op, PatternRewriter& rewriter) const {
        auto input = op.getInput();
        Operation* defInput = input.getDefiningOp();

        auto dict = defInput->getAttrDictionary();
        if (!dict) return failure();

        const BandedSubMatrix inputBand = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        const uint64_t lower = inputBand.Property.LowerBandwidth;
        const uint64_t upper = inputBand.Property.UpperBandwidth;

        auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
        const int64_t N = resultType.getDimSize(0);
        const int64_t M = resultType.getDimSize(1);

        Location loc = op->getLoc();

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value dimN = arith::ConstantIndexOp::create(rewriter, loc, N);
        Value dimM = arith::ConstantIndexOp::create(rewriter, loc, M);

        Value lowerBW = arith::ConstantIndexOp::create(rewriter, loc, (int64_t)lower);
        Value upperBW = arith::ConstantIndexOp::create(rewriter, loc, (int64_t)upper);

        // result
        Value emptyTensor = tensor::EmptyOp::create(rewriter, loc, resultType, ValueRange{});
        auto elementType = resultType.getElementType();
        Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                               rewriter.getZeroAttr(elementType));
        Value result =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ op.getInit() })
                .getResult(0);

        auto iLoop = scf::ForOp::create(
            rewriter, loc, c0, dimN, c1, ValueRange{ result },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                Value cOut = iArgs[0];
                // jStart = max(0, i - L)
                Value lMinusI = arith::SubIOp::create(ob, loc, i, lowerBW);
                Value jStart = arith::MaxSIOp::create(ob, loc, lMinusI, c0);
                // end = min(i + U + 1, M)
                Value iPlusU = arith::AddIOp::create(ob, loc, i, upperBW);
                Value iPlusUPlusOne =
                    arith::AddIOp::create(ob, loc, arith::AddIOp::create(ob, loc, i, upperBW), c1);
                Value jEnd = arith::MinSIOp::create(ob, loc, iPlusUPlusOne, dimM);
                auto jLoop = scf::ForOp::create(
                    ob, loc, jStart, jEnd, c1, ValueRange{ cOut },
                    [&](OpBuilder& ib, Location loc, Value j, ValueRange jArgs) {
                        Value cIn = jArgs[0];
                        Value val = tensor::ExtractOp::create(ib, loc, input, ValueRange{ i, j });
                        Value updated =
                            tensor::InsertOp::create(ib, loc, val, cIn, ValueRange{ j, i });
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    });
                scf::YieldOp::create(ob, loc, jLoop.getResults());
            });
        iLoop->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, iLoop.getResult(0));
        return success();
    }

    LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                  PatternRewriter& rewriter) const override {
        // TODO: check for permutation.
        auto dict = op->getAttrDictionary();
        if (!dict) dict = DictionaryAttr();
        BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        if (opBandInfo.isDiagonal()) {
            rewriter.replaceOp(op, op.getInput());
            return success();
        }
        return denseBandedTranspose(op, rewriter);
    }
};

struct BandedRewrite : public impl::BandedRewriteBase<BandedRewrite> {
    using BandedRewriteBase::BandedRewriteBase;

    void runOnOperation() override {
#ifdef ENABLE_BENCHMARKING
        auto start{ std::chrono::high_resolution_clock::now() };
#endif
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = funcOp.getContext();

        // check if the analysis propagated dia flags
        auto cached = getCachedAnalysis<BandedAnalysisResult>();
        bool detectDIA = cached ? cached->get().detectDIA : false;

        RewritePatternSet patterns(context);

        patterns
            .add<MatMulPattern, GenericElementWisePattern, TransposePattern, BatchMatmulPattern>(
                context, detectDIA);

        addDIAElementwisePatterns(patterns);
        addDIABatchMatmulPatterns(patterns);
        addDIAMatmulPatterns(patterns, detectDIA);
        addDIATransposePatterns(patterns);
        addDIASoftmaxPatterns(patterns);

        GreedyRewriteConfig config;
        config.setMaxIterations(1);
        config.setUseTopDownTraversal(true);

        (void)applyPatternsGreedily(funcOp, std::move(patterns), config);
#ifdef ENABLE_BENCHMARKING
        auto end{ std::chrono::high_resolution_clock::now() };
        llvm::errs() << "BandedRewrite time: "
                     << std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end -
                                                                                              start)
                            .count()
                     << " ms\n";
#endif
    }
};
}  // namespace mlir::bpa
