#include "Transform/BandedRewrite.h"

#include <cstdint>
#include <optional>

#include "Analysis/BandedStructureAnalysis.h"
#include "Dialect/DIA/DIAOps.h"
#include "Transform/BandedPropagation.h"
#include "Utils/TransformUtils.h"
#include "llvm/ADT/SmallVector.h"
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

struct DIATransposePattern : public OpRewritePattern<dia::TransposeOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(dia::TransposeOp op, PatternRewriter& rewriter) const override {
        auto dict = op->getAttrDictionary();
        if (!dict) dict = DictionaryAttr();

        const BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        if (opBandInfo.isDiagonal()) {
            rewriter.replaceOp(op, op.getInput());
            return success();
        } else
            return diaBandedTranspose(op, rewriter);
    }

    LogicalResult diaBandedTranspose(dia::TransposeOp op, PatternRewriter& rewriter) const {
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
        Value dimM = arith::ConstantIndexOp::create(rewriter, loc, M);

        Value lowerBW = arith::ConstantIndexOp::create(rewriter, loc, (int64_t)lower);
        Value upperBW = arith::ConstantIndexOp::create(rewriter, loc, (int64_t)upper);

        Value K =
            arith::ConstantIndexOp::create(rewriter, loc, static_cast<int64_t>(lower + upper + 1));
        Value uPlusL =
            arith::ConstantIndexOp::create(rewriter, loc, static_cast<int64_t>(lower + upper));

        // result
        Value emptyTensor = tensor::EmptyOp::create(rewriter, loc, resultType, ValueRange{});
        auto elementType = resultType.getElementType();
        Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                               rewriter.getZeroAttr(elementType));
        Value zeroedC =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ emptyTensor })
                .getResult(0);

        auto iLoop = scf::ForOp::create(
            rewriter, loc, c0, K, c1, ValueRange{ zeroedC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                Value cOut = iArgs[0];
                // jStart = max(0, l - i)
                Value lMinusI = arith::SubIOp::create(ob, loc, lowerBW, i);
                Value jStart = arith::MaxSIOp::create(ob, loc, lMinusI, c0);
                // end = M - max(0, i - l)
                Value iMinusL = arith::SubIOp::create(ob, loc, i, lowerBW);
                Value iMinusLClamped = arith::MaxSIOp::create(ob, loc, iMinusL, c0);
                Value jEnd = arith::SubIOp::create(ob, loc, dimM, iMinusLClamped);
                // ni = u + l - i
                Value newI = arith::SubIOp::create(ob, loc, uPlusL, i);
                auto jLoop = scf::ForOp::create(
                    ob, loc, jStart, jEnd, c1, ValueRange{ cOut },
                    [&](OpBuilder& ib, Location loc, Value j, ValueRange jArgs) {
                        Value cIn = jArgs[0];
                        // nj = j + iMinusL
                        Value newJ = arith::AddIOp::create(ib, loc, iMinusL, j);
                        Value val = tensor::ExtractOp::create(ib, loc, input, ValueRange{ i, j });
                        Value updated =
                            tensor::InsertOp::create(ib, loc, val, cIn, ValueRange{ newI, newJ });
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    });
                scf::YieldOp::create(ob, loc, jLoop.getResults());
            });
        rewriter.replaceOp(op, iLoop.getResult(0));
        return success();
    }
};
// ------------------------------------------------------------------------------------------------------------------------------
// dia.BatchMatmulOp
// ------------------------------------------------------------------------------------------------------------------------------

struct DIABatchMatMulPattern : public OpRewritePattern<dia::BatchMatmulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult diaTimesDiaToDiaBandedBatchMatmulToSCF(dia::BatchMatmulOp op,
                                                         PatternRewriter& rewriter,
                                                         const BandedSubMatrix& bandResult) const {
        Location loc = op.getLoc();
        Value A = op.getLhs();
        Value B = op.getRhs();
        Value C = op.getOutput();

        Operation* defOpA = A.getDefiningOp();
        Operation* defOpB = B.getDefiningOp();

        auto dictA = defOpA->getAttrDictionary();
        auto dictB = defOpB->getAttrDictionary();

        if (!dictA || !dictB) return failure();

        const BandedSubMatrix bandA = BandedStructureAnalysis::readPropertyFromDictAttr(dictA);
        const BandedSubMatrix bandB = BandedStructureAnalysis::readPropertyFromDictAttr(dictB);

        auto resultType = cast<RankedTensorType>(A.getType());
        const int64_t batch{ resultType.getDimSize(0) };
        const int64_t N{ resultType.getDimSize(2) };

        const int64_t lA = bandA.Property.LowerBandwidth;
        const int64_t uA = bandA.Property.UpperBandwidth;
        const int64_t lB = bandB.Property.LowerBandwidth;
        const int64_t uB = bandB.Property.UpperBandwidth;
        const int64_t lC = bandResult.Property.LowerBandwidth;
        const int64_t uC = bandResult.Property.UpperBandwidth;

        auto elementType = cast<RankedTensorType>(A.getType()).getElementType();

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value dimN = arith::ConstantIndexOp::create(rewriter, loc, N);
        Value batchVal = arith::ConstantIndexOp::create(rewriter, loc, batch);

        Value lAVal = arith::ConstantIndexOp::create(rewriter, loc, lA);
        Value uAVal = arith::ConstantIndexOp::create(rewriter, loc, uA);
        Value lBVal = arith::ConstantIndexOp::create(rewriter, loc, lB);
        Value uBVal = arith::ConstantIndexOp::create(rewriter, loc, uB);
        Value lCVal = arith::ConstantIndexOp::create(rewriter, loc, lC);
        Value uCVal = arith::ConstantIndexOp::create(rewriter, loc, uC);

        // negative constants for lower bounds
        Value negLA = arith::ConstantIndexOp::create(rewriter, loc, -lA);
        Value negLB = arith::ConstantIndexOp::create(rewriter, loc, -lB);
        Value negLC = arith::ConstantIndexOp::create(rewriter, loc, -lC);

        Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                               rewriter.getZeroAttr(elementType));
        Value zeroedC{
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0)
        };

        // outer loop: for i in range(-lc, uc + 1)
        Value iStart = negLC;
        Value iEnd = arith::AddIOp::create(rewriter, loc, uCVal, c1);

        // outer loop: for i_shifted in range(0, lC + uC + 1)
        Value totalDiags = arith::ConstantIndexOp::create(rewriter, loc, lC + uC + 1);

        auto csti64 = [&](int64_t val) {
            return arith::ConstantIntOp::create(rewriter, loc, val, 64);
        };
        auto toIndex = [&](OpBuilder& b, Value v) {
            return arith::IndexCastOp::create(b, loc, rewriter.getIndexType(), v);
        };
        auto toI64 = [&](OpBuilder& b, Value v) {
            return arith::IndexCastOp::create(b, loc, rewriter.getIntegerType(64), v);
        };

        // constants as i64
        Value lAi64 = csti64(lA);
        Value uAi64 = csti64(uA);
        Value lBi64 = csti64(lB);
        Value uBi64 = csti64(uB);
        Value lCi64 = csti64(lC);
        Value Ni64 = csti64(N);
        Value c0i64 = csti64(0);
        Value c1i64 = csti64(1);

        auto bLoop{ scf::ForOp::create(
            rewriter, loc, c0, batchVal, c1, ValueRange{ zeroedC },
            [&](OpBuilder& bb, Location loc, Value batchIdx, ValueRange bArgs) {
                Value cBatch{ bArgs[0] };
                auto iLoop = scf::ForOp::create(
                    rewriter, loc, c0, totalDiags, c1, ValueRange{ cBatch },
                    [&](OpBuilder& ob, Location loc, Value iShifted, ValueRange iArgs) {
                        Value cOuter = iArgs[0];

                        // i = iShifted - lC, computed in i64
                        Value iShiftedI64 = toI64(ob, iShifted);
                        Value i = arith::SubIOp::create(ob, loc, iShiftedI64, lCi64);

                        // cIdx = iShifted
                        Value cIdx = iShifted;

                        // j_start = max(-lA, i - uB) in i64
                        Value negLA = csti64(-lA);
                        Value iMinusUb = arith::SubIOp::create(ob, loc, i, uBi64);
                        Value jStartI64 = arith::MaxSIOp::create(ob, loc, negLA, iMinusUb);

                        // j_end = min(uA, i + lB) + 1 in i64
                        Value iPlusLb = arith::AddIOp::create(ob, loc, i, lBi64);
                        Value jEndI64 = arith::AddIOp::create(
                            ob, loc, arith::MinSIOp::create(ob, loc, uAi64, iPlusLb), c1i64);

                        // shift j to index: j_shifted_start = j_start + lA
                        Value jStartShifted =
                            toIndex(ob, arith::AddIOp::create(ob, loc, jStartI64, lAi64));
                        Value jEndShifted =
                            toIndex(ob, arith::AddIOp::create(ob, loc, jEndI64, lAi64));

                        auto jLoop = scf::ForOp::create(
                            ob, loc, jStartShifted, jEndShifted, c1, ValueRange{ cOuter },
                            [&](OpBuilder& mb, Location loc, Value jShifted, ValueRange jArgs) {
                                Value cMid = jArgs[0];

                                // j = jShifted - lA in i64
                                Value jShiftedI64 = toI64(mb, jShifted);
                                Value j = arith::SubIOp::create(mb, loc, jShiftedI64, lAi64);

                                // aIdx = jShifted
                                Value aIdx = jShifted;

                                // k = i - j in i64, bIdx = k + lB
                                Value k = arith::SubIOp::create(mb, loc, i, j);
                                Value bIdx = toIndex(mb, arith::AddIOp::create(mb, loc, k, lBi64));

                                // r_start = max(0, -j) in i64
                                Value negJ = arith::SubIOp::create(mb, loc, c0i64, j);
                                Value rStart =
                                    toIndex(mb, arith::MaxSIOp::create(mb, loc, c0i64, negJ));

                                // r_end = min(N, N - j) in i64
                                Value nMinusJ = arith::SubIOp::create(mb, loc, Ni64, j);
                                Value rEnd =
                                    toIndex(mb, arith::MinSIOp::create(mb, loc, Ni64, nMinusJ));

                                auto rLoop = scf::ForOp::create(
                                    mb, loc, rStart, rEnd, c1, ValueRange{ cMid },
                                    [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                                        Value cInner = rArgs[0];

                                        Value cVal = tensor::ExtractOp::create(
                                            ib, loc, cInner, ValueRange{ batchIdx, cIdx, r });
                                        Value aVal = tensor::ExtractOp::create(
                                            ib, loc, A, ValueRange{ batchIdx, aIdx, r });

                                        // r + j in i64, then cast to index
                                        Value rI64 = toI64(ib, r);
                                        Value rPlusJ =
                                            toIndex(ib, arith::AddIOp::create(ib, loc, rI64, j));
                                        Value bVal = tensor::ExtractOp::create(
                                            ib, loc, B, ValueRange{ batchIdx, bIdx, rPlusJ });

                                        Value mul = arith::MulFOp::create(ib, loc, aVal, bVal);
                                        Value acc = arith::AddFOp::create(ib, loc, cVal, mul);
                                        Value updated = tensor::InsertOp::create(
                                            ib, loc, acc, cInner, ValueRange{ batchIdx, cIdx, r });

                                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                                    });
                                scf::YieldOp::create(mb, loc, rLoop.getResults());
                            });
                        scf::YieldOp::create(ob, loc, jLoop.getResults());
                    });
                scf::YieldOp::create(bb, loc, iLoop.getResults());
            }) };
        bLoop->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, bLoop.getResult(0));
        return success();
    }

    LogicalResult matchAndRewrite(dia::BatchMatmulOp op, PatternRewriter& rewriter) const override {
        auto dict = op->getAttrDictionary();
        if (!dict) dict = DictionaryAttr();

        const BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        if (opBandInfo.isDiagonal()) return failure();
        // banded
        else
            return diaTimesDiaToDiaBandedBatchMatmulToSCF(op, rewriter, opBandInfo);
    }
};

// ------------------------------------------------------------------------------------------------------------------------------
// dia.ElementwiseOp
// ------------------------------------------------------------------------------------------------------------------------------

struct DIAElementwisePattern : public OpRewritePattern<dia::ElementwiseOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult diaTimesDiaToDenseBandedElementwiseToSCF(
        dia::ElementwiseOp op, PatternRewriter& rewriter, const BandedSubMatrix& bandResult) const {
        assert(op.getKind() != dia::ElementwiseKind::mul);

        Location loc{ op.getLoc() };
        Value A{ op.getInputs()[0] };
        Value B{ op.getInputs()[1] };
        Value C{ op.getOutput() };

        auto outputType{ cast<RankedTensorType>(C.getType()) };

        auto elementType{ cast<RankedTensorType>(A.getType()).getElementType() };

        MLIRContext* context{ rewriter.getContext() };

        Operation* defOpA{ A.getDefiningOp() };
        Operation* defOpB{ B.getDefiningOp() };
        auto dictA{ defOpA->getAttrDictionary() };
        auto dictB{ defOpB->getAttrDictionary() };
        BandedSubMatrix bandA{ BandedStructureAnalysis::readPropertyFromDictAttr(dictA) };
        BandedSubMatrix bandB{ BandedStructureAnalysis::readPropertyFromDictAttr(dictB) };

        auto lA{ bandA.Property.LowerBandwidth };
        auto uA{ bandA.Property.UpperBandwidth };
        auto lB{ bandB.Property.LowerBandwidth };
        auto uB{ bandB.Property.UpperBandwidth };

        Value zero{ arith::ConstantOp::create(rewriter, loc, elementType,
                                              rewriter.getZeroAttr(elementType)) };
        Value zeroedC{
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0)
        };

        auto c0{ arith::ConstantIndexOp::create(rewriter, loc, 0) };
        auto c1{ arith::ConstantIndexOp::create(rewriter, loc, 1) };
        auto cf0{ arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getFloatAttr(elementType, 0.0)) };

        auto rank{ outputType.getRank() };

        auto totalRows{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 2)) };
        auto totalCols{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 1)) };

        auto lDiff{ arith::ConstantIndexOp::create(rewriter, loc,
                                                   std::max(lA, lB) - std::min(lA, lB)) };
        auto lMax{ arith::ConstantIndexOp::create(rewriter, loc, std::max(lA, lB)) };
        auto lMin{ arith::ConstantIndexOp::create(rewriter, loc, std::min(lA, lB)) };
        auto uDiff{ arith::ConstantIndexOp::create(rewriter, loc,
                                                   std::max(uA, uB) - std::min(uA, uB)) };
        auto uMin{ arith::ConstantIndexOp::create(rewriter, loc, std::min(uA, uB)) };

        auto cLA{ arith::ConstantIndexOp::create(rewriter, loc, lA) };
        auto cLB{ arith::ConstantIndexOp::create(rewriter, loc, lB) };

        auto currC{ zeroedC };

        if (lA > lB || lB > lA) {
            scf::ForOp iLoop{ scf::ForOp::create(
                rewriter, loc, c0, lDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::SubIOp::create(ob, loc, lMax, i) };
                    Value mMinusBand{ arith::SubIOp::create(ob, loc, totalRows, currBand) };
                    Value numEls{ arith::MinUIOp::create(ob, loc, mMinusBand, totalCols) };
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value operand1, operand2;

                            if (lA > lB) {
                                operand1 =
                                    tensor::ExtractOp::create(ib, loc, elementType, A, { i, r });
                                operand2 = cf0;
                            } else {
                                operand1 = cf0;
                                operand2 =
                                    tensor::ExtractOp::create(ib, loc, elementType, B, { i, r });
                            }

                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2);
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2);
                                    break;
                                default:
                                    assert(false);
                            }
                            Value cRow{ arith::AddIOp::create(ib, loc, r, currBand) };
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ cRow, r }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
            currC = iLoop.getResult(0);
        }

        if (uA > uB || uB > uA) {
            auto startUpperBand{ arith::ConstantIndexOp::create(rewriter, loc,
                                                                std::min(uA, uB) + 1) };
            scf::ForOp iLoop{ scf::ForOp::create(
                rewriter, loc, c0, uDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::AddIOp::create(ob, loc, i, startUpperBand) };
                    Value aInd, bInd;
                    if (uA > uB) {
                        aInd = arith::AddIOp::create(ob, loc, currBand, cLA);
                    } else {
                        bInd = arith::AddIOp::create(ob, loc, currBand, cLB);
                    }
                    Value nMinusBand{ arith::SubIOp::create(ob, loc, totalCols, currBand) };
                    Value numEls{ arith::MinUIOp::create(ob, loc, totalRows, nMinusBand) };
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value operand1, operand2;
                            Value shiftedR{ arith::AddIOp::create(ib, loc, r, currBand) };
                            if (uA > uB) {
                                operand1 = tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                     { aInd, shiftedR });
                                operand2 = cf0;
                            } else {
                                operand1 = cf0;
                                operand2 = tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                     { bInd, shiftedR });
                            }

                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2);
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2);
                                    break;
                                default:
                                    assert(false);
                            }

                            Value cInd{ arith::AddIOp::create(ib, loc, r, currBand) };
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   { r, cInd }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
            currC = iLoop.getResult(0);
        }

        auto totalLowerDiags{ arith::ConstantIndexOp::create(rewriter, loc, std::min(lA, lB) + 1) };
        scf::ForOp iLoop{ scf::ForOp::create(
            rewriter, loc, c0, totalLowerDiags, c1, ValueRange{ currC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                auto cOuter{ iArgs[0] };
                Value aInd{ (lA > lB) ? arith::AddIOp::create(ob, loc, i, lDiff) : i };
                Value bInd{ (lA > lB) ? i : arith::AddIOp::create(ob, loc, i, lDiff) };

                Value currBand{ arith::SubIOp::create(ob, loc, lMin, i) };
                Value mMinusBand{ arith::SubIOp::create(ob, loc, totalRows, currBand) };
                Value numEls{ arith::MinUIOp::create(ob, loc, mMinusBand, totalCols) };
                auto rLoop{ scf::ForOp::create(
                    ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                    [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                        auto cInner{ rArgs[0] };
                        auto operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                 { aInd, r }) };
                        auto operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                 { bInd, r }) };

                        Value newOp;
                        switch (op.getKind()) {
                            case dia::ElementwiseKind::add:
                                newOp = arith::AddFOp::create(ib, loc, operand1, operand2);
                                break;
                            case dia::ElementwiseKind::sub:
                                newOp = arith::SubFOp::create(ib, loc, operand1, operand2);
                                break;
                            default:
                                assert(false);
                        }

                        Value cRow{ arith::AddIOp::create(ib, loc, r, currBand) };
                        auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                               ValueRange{ cRow, r }) };
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    }) };
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            }) };
        currC = iLoop.getResult(0);

        if (std::min(uA, uB) != 0) {
            auto totalUpperDiags{ arith::ConstantIndexOp::create(rewriter, loc,
                                                                 std::min(uA, uB) + 1) };
            auto iLoop{ scf::ForOp::create(
                rewriter, loc, c1, totalUpperDiags, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value aInd{ arith::AddIOp::create(ob, loc, i, cLA) };
                    Value bInd{ arith::AddIOp::create(ob, loc, i, cLB) };
                    Value nMinusBand{ arith::SubIOp::create(ob, loc, totalCols, i) };
                    Value numEls{ arith::MinUIOp::create(ob, loc, totalRows, nMinusBand) };
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            Value shiftedR{ arith::AddIOp::create(ib, loc, r, i) };
                            auto cInner{ rArgs[0] };
                            auto operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                     { aInd, shiftedR }) };
                            auto operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                     { bInd, shiftedR }) };

                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2);
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2);
                                    break;
                                default:
                                    assert(false);
                            }

                            Value cCol{ arith::AddIOp::create(ib, loc, r, i) };
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ r, cCol }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
            currC = iLoop.getResult(0);
        }

        currC.getDefiningOp()->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, currC);

        return success();
    }

<<<<<<< HEAD
    LogicalResult diaTimesDiaToDiaBandedElementwiseToSCF(dia::ElementwiseOp op, PatternRewriter& rewriter,
                                                    const BandedSubMatrix& bandResult) const {
=======
    LogicalResult diaTimesDiaToDiaBandedElementwiseToSCF(dia::ElementwiseOp op,
                                                         PatternRewriter& rewriter,
                                                         const BandedSubMatrix& bandResult) const {
>>>>>>> eef54b24785336690504d4cf77753478ae05978b
        Location loc{ op.getLoc() };
        Value A{ op.getInputs()[0] };
        Value B{ op.getInputs()[1] };
        Value C{ op.getOutput() };

        auto elementType{ cast<RankedTensorType>(A.getType()).getElementType() };
        MLIRContext* context{ rewriter.getContext() };

        BandedSubMatrix bandA{ BandedStructureAnalysis::readPropertyFromDictAttr(
            A.getDefiningOp()->getAttrDictionary()) };
        BandedSubMatrix bandB{ BandedStructureAnalysis::readPropertyFromDictAttr(
            B.getDefiningOp()->getAttrDictionary()) };

        auto lA{ bandA.Property.LowerBandwidth };
        auto uA{ bandA.Property.UpperBandwidth };
        auto lB{ bandB.Property.LowerBandwidth };
        auto uB{ bandB.Property.UpperBandwidth };
        auto lC{ bandResult.Property.LowerBandwidth };

        Value zero{ arith::ConstantOp::create(rewriter, loc, elementType,
                                              rewriter.getZeroAttr(elementType)) };
        Value zeroedC{
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0)
        };

        auto c0{ arith::ConstantIndexOp::create(rewriter, loc, 0) };
        auto c1{ arith::ConstantIndexOp::create(rewriter, loc, 1) };
        auto cf0{ arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getFloatAttr(elementType, 0.0)) };

        auto resultType{ cast<RankedTensorType>(A.getType()) };
        auto totalCols{ arith::ConstantIndexOp::create(
            rewriter, loc, resultType.getDimSize(resultType.getRank() - 1)) };

        auto cLA{ arith::ConstantIndexOp::create(rewriter, loc, lA) };
        auto cLB{ arith::ConstantIndexOp::create(rewriter, loc, lB) };
        auto cLC{ arith::ConstantIndexOp::create(rewriter, loc, lC) };

        auto currC{ zeroedC };

        if (op.getKind() != dia::ElementwiseKind::mul && (lA > lB || lB > lA)) {
            auto totalDiags{ arith::ConstantIndexOp::create(rewriter, loc,
                                                            std::max(lA, lB) - std::min(lA, lB)) };
            auto lMax{ arith::ConstantIndexOp::create(rewriter, loc, std::max(lA, lB)) };

            scf::ForOp iLoop{ scf::ForOp::create(
                rewriter, loc, c0, totalDiags, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::SubIOp::create(ob, loc, i, lMax).getResult() };
                    Value cInd{ arith::AddIOp::create(ob, loc, cLC, currBand).getResult() };

                    Value aInd, bInd;
                    if (lA > lB)
                        aInd = arith::AddIOp::create(ob, loc, cLA, currBand).getResult();
                    else
                        bInd = arith::AddIOp::create(ob, loc, cLB, currBand).getResult();

                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, totalCols, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            Value operand1{ lA > lB
                                                ? tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                            ValueRange{ aInd, r })
                                                      .getResult()
                                                : cf0 };
                            Value operand2{ lA > lB
                                                ? cf0
                                                : tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                            ValueRange{ bInd, r })
                                                      .getResult() };

                            Value newOp;
                            if (op.getKind() == dia::ElementwiseKind::add)
                                newOp =
                                    arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                            else
                                newOp =
                                    arith::SubFOp::create(ib, loc, operand1, operand2).getResult();

                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, rArgs[0],
                                                                   ValueRange{ cInd, r }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
            currC = iLoop.getResult(0);
        }

        if (op.getKind() != dia::ElementwiseKind::mul && (uA > uB || uB > uA)) {
            auto numDiagonals{ arith::ConstantIndexOp::create(
                rewriter, loc, std::max(uA, uB) - std::min(uA, uB)) };
            auto startUpperBand{ arith::ConstantIndexOp::create(rewriter, loc,
                                                                std::min(uA, uB) + 1) };

            scf::ForOp iLoop{ scf::ForOp::create(
                rewriter, loc, c0, numDiagonals, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::AddIOp::create(ob, loc, i, startUpperBand).getResult() };
                    Value cInd{ arith::AddIOp::create(ob, loc, cLC, currBand).getResult() };

                    Value aInd, bInd;
                    if (uA > uB)
                        aInd = arith::AddIOp::create(ob, loc, cLA, currBand).getResult();
                    else
                        bInd = arith::AddIOp::create(ob, loc, cLB, currBand).getResult();

                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, totalCols, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            Value operand1{ uA > uB
                                                ? tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                            ValueRange{ aInd, r })
                                                      .getResult()
                                                : cf0 };
                            Value operand2{ uA > uB
                                                ? cf0
                                                : tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                            ValueRange{ bInd, r })
                                                      .getResult() };

                            Value newOp;
                            if (op.getKind() == dia::ElementwiseKind::add)
                                newOp =
                                    arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                            else
                                newOp =
                                    arith::SubFOp::create(ib, loc, operand1, operand2).getResult();

                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, rArgs[0],
                                                                   ValueRange{ cInd, r }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
            currC = iLoop.getResult(0);
        }
<<<<<<< HEAD
        
        auto numDiagonals{ arith::ConstantIndexOp::create(rewriter, loc, std::min(lA, lB) + 1 + std::min(uA, uB)) };
=======

        // Intersecting Bands (Handles Add, Sub, and Mul)
        auto numDiagonals{ arith::ConstantIndexOp::create(
            rewriter, loc, std::min(lA, lB) + 1 + std::min(uA, uB)) };
>>>>>>> eef54b24785336690504d4cf77753478ae05978b
        auto minL{ arith::ConstantIndexOp::create(rewriter, loc, std::min(lA, lB)) };

        scf::ForOp iLoop{ scf::ForOp::create(
            rewriter, loc, c0, numDiagonals, c1, ValueRange{ currC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                auto cOuter{ iArgs[0] };
                Value currBand{ arith::SubIOp::create(ob, loc, i, minL).getResult() };
                Value aInd{ arith::AddIOp::create(ob, loc, cLA, currBand).getResult() };
                Value bInd{ arith::AddIOp::create(ob, loc, cLB, currBand).getResult() };
                Value cInd{ arith::AddIOp::create(ob, loc, cLC, currBand).getResult() };

                auto rLoop{ scf::ForOp::create(
                    ob, loc, c0, totalCols, c1, ValueRange{ cOuter },
                    [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                        Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                  ValueRange{ aInd, r }) };
                        Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                  ValueRange{ bInd, r }) };

                        Value newOp;
                        switch (op.getKind()) {
                            case dia::ElementwiseKind::add:
                                newOp =
                                    arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            case dia::ElementwiseKind::sub:
                                newOp =
                                    arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            case dia::ElementwiseKind::mul:
                                newOp =
                                    arith::MulFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            default:
                                assert(false);
                        }

                        auto updated{ tensor::InsertOp::create(ib, loc, newOp, rArgs[0],
                                                               ValueRange{ cInd, r }) };
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    }) };
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            }) };

        currC = iLoop.getResult(0);

        currC.getDefiningOp()->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, currC);

        return success();
    }

<<<<<<< HEAD
    LogicalResult diaTimesDenseToDiaBandedElementwiseToSCF(dia::ElementwiseOp op, PatternRewriter& rewriter,
                                                           const BandedSubMatrix& bandResult) const {
        assert(op.getKind() == dia::ElementwiseKind::mul);

        Location loc{ op.getLoc() };
        Value A{ op.getInputs()[0] };
        Value B{ op.getInputs()[1] };
        Value C{ op.getOutput() };

        auto elementType{ cast<RankedTensorType>(A.getType()).getElementType() };
        MLIRContext* context{ rewriter.getContext() };

        BandedSubMatrix bandA{ BandedStructureAnalysis::readPropertyFromDictAttr(A.getDefiningOp()->getAttrDictionary()) };

        auto lA{ bandA.Property.LowerBandwidth };
        auto lC{ bandResult.Property.LowerBandwidth };
        auto uC{ bandResult.Property.UpperBandwidth };

        Value zero{ arith::ConstantOp::create(rewriter, loc, elementType, rewriter.getZeroAttr(elementType)) };
        Value zeroedC{ linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0) };

        auto c0{ arith::ConstantIndexOp::create(rewriter, loc, 0) };
        auto c1{ arith::ConstantIndexOp::create(rewriter, loc, 1) };

        auto outputType{ cast<RankedTensorType>(B.getType()) }; 
        auto rank{ outputType.getRank() };
        auto totalRows{ arith::ConstantIndexOp::create(rewriter, loc, outputType.getDimSize(rank - 2)) };
        auto totalCols{ arith::ConstantIndexOp::create(rewriter, loc, outputType.getDimSize(rank - 1)) };

        auto cLA{ arith::ConstantIndexOp::create(rewriter, loc, lA) };
        auto cLC{ arith::ConstantIndexOp::create(rewriter, loc, lC) };

        auto currC{ zeroedC };

        if (lC > 0) {
            auto totalLowerDiags{ arith::ConstantIndexOp::create(rewriter, loc, lC) };
            scf::ForOp iLoopLower{ scf::ForOp::create(
                rewriter, loc, c0, totalLowerDiags, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::SubIOp::create(ob, loc, cLC, i).getResult() };
                    Value aInd{ arith::SubIOp::create(ob, loc, cLA, currBand).getResult() };
                    
                    Value mMinusBand{ arith::SubIOp::create(ob, loc, totalRows, currBand).getResult() };
                    Value numEls{ arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult() };
                    
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cRow{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            
                            Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A, ValueRange{ aInd, r }).getResult() };
                            Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B, ValueRange{ cRow, r }).getResult() };
                            
                            Value newOp{ arith::MulFOp::create(ib, loc, operand1, operand2).getResult() };
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner, ValueRange{ i, r }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        })};
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                })};
            currC = iLoopLower.getResult(0);
        }

        auto totalUpperDiags{ arith::ConstantIndexOp::create(rewriter, loc, uC + 1) };
        scf::ForOp iLoopUpper{ scf::ForOp::create(
            rewriter, loc, c0, totalUpperDiags, c1, ValueRange{ currC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                auto cOuter{ iArgs[0] };
                Value currBand{ i };
                Value aInd{ arith::AddIOp::create(ob, loc, cLA, currBand).getResult() };
                Value cInd{ arith::AddIOp::create(ob, loc, cLC, currBand).getResult() };
                
                Value nMinusBand{ arith::SubIOp::create(ob, loc, totalCols, currBand).getResult() };
                Value numEls{ arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult() };
                
                auto rLoop{ scf::ForOp::create(
                    ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                    [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                        auto cInner{ rArgs[0] };
                        Value cCol{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                        
                        Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A, ValueRange{ aInd, r }).getResult() };
                        Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B, ValueRange{ r, cCol }).getResult() };
                        
                        Value newOp{ arith::MulFOp::create(ib, loc, operand1, operand2).getResult() };
                        auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner, ValueRange{ cInd, r }) };
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    })};
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            })};
        currC = iLoopUpper.getResult(0);

        currC.getDefiningOp()->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, currC);
        return success();
    }


    LogicalResult denseTimesDiaToDiaBandedElementwiseToSCF(dia::ElementwiseOp op, PatternRewriter& rewriter,
                                                           const BandedSubMatrix& bandResult) const {
        assert(op.getKind() == dia::ElementwiseKind::mul);

        Location loc{ op.getLoc() };
        Value A{ op.getInputs()[0] };
        Value B{ op.getInputs()[1] };
        Value C{ op.getOutput() };

        auto elementType{ cast<RankedTensorType>(A.getType()).getElementType() };
        MLIRContext* context{ rewriter.getContext() };

        BandedSubMatrix bandB{ BandedStructureAnalysis::readPropertyFromDictAttr(B.getDefiningOp()->getAttrDictionary()) };

        auto lB{ bandB.Property.LowerBandwidth };
        auto lC{ bandResult.Property.LowerBandwidth };
        auto uC{ bandResult.Property.UpperBandwidth };

        Value zero{ arith::ConstantOp::create(rewriter, loc, elementType, rewriter.getZeroAttr(elementType)) };
        Value zeroedC{ linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0) };

        auto c0{ arith::ConstantIndexOp::create(rewriter, loc, 0) };
        auto c1{ arith::ConstantIndexOp::create(rewriter, loc, 1) };

        auto outputType{ cast<RankedTensorType>(A.getType()) };
        auto rank{ outputType.getRank() };
        auto totalRows{ arith::ConstantIndexOp::create(rewriter, loc, outputType.getDimSize(rank - 2)) };
        auto totalCols{ arith::ConstantIndexOp::create(rewriter, loc, outputType.getDimSize(rank - 1)) };

        auto cLB{ arith::ConstantIndexOp::create(rewriter, loc, lB) };
        auto cLC{ arith::ConstantIndexOp::create(rewriter, loc, lC) };

        auto currC{ zeroedC };

        if (lC > 0) {
            auto totalLowerDiags{ arith::ConstantIndexOp::create(rewriter, loc, lC) };
            scf::ForOp iLoopLower{ scf::ForOp::create(
                rewriter, loc, c0, totalLowerDiags, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::SubIOp::create(ob, loc, cLC, i).getResult() };
                    Value bInd{ arith::SubIOp::create(ob, loc, cLB, currBand).getResult() };
                    
                    Value mMinusBand{ arith::SubIOp::create(ob, loc, totalRows, currBand).getResult() };
                    Value numEls{ arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult() };
                    
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cRow{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            
                            Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A, ValueRange{ cRow, r }).getResult() };
                            Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B, ValueRange{ bInd, r }).getResult() };
                            
                            Value newOp{ arith::MulFOp::create(ib, loc, operand1, operand2).getResult() };
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner, ValueRange{ i, r }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        })};
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                })};
            currC = iLoopLower.getResult(0);
        }

        auto totalUpperDiags{ arith::ConstantIndexOp::create(rewriter, loc, uC + 1) };
        scf::ForOp iLoopUpper{ scf::ForOp::create(
            rewriter, loc, c0, totalUpperDiags, c1, ValueRange{ currC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                auto cOuter{ iArgs[0] };
                Value currBand{ i };
                Value bInd{ arith::AddIOp::create(ob, loc, cLB, currBand).getResult() };
                Value cInd{ arith::AddIOp::create(ob, loc, cLC, currBand).getResult() };
                
                Value nMinusBand{ arith::SubIOp::create(ob, loc, totalCols, currBand).getResult() };
                Value numEls{ arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult() };
                
                auto rLoop{ scf::ForOp::create(
                    ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                    [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                        auto cInner{ rArgs[0] };
                        Value cCol{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                        
                        Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A, ValueRange{ r, cCol }).getResult() };
                        Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B, ValueRange{ bInd, r }).getResult() };
                        
                        Value newOp{ arith::MulFOp::create(ib, loc, operand1, operand2).getResult() };
                        auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner, ValueRange{ cInd, r }) };
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    })};
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            })};
        currC = iLoopUpper.getResult(0);

        currC.getDefiningOp()->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, currC);
        return success();
    }

    LogicalResult diaToDiaBandedElementwiseToLinalg(dia::ElementwiseOp op, PatternRewriter& rewriter,
=======
    LogicalResult diaToDiaBandedElementwiseToLinalg(dia::ElementwiseOp op,
                                                    PatternRewriter& rewriter,
>>>>>>> eef54b24785336690504d4cf77753478ae05978b
                                                    const BandedSubMatrix& bandResult) const {
        Location loc{ op.getLoc() };
        Value input{ op.getInputs()[0] };
        Value output{ op.getOutput() };

        MLIRContext* context{ rewriter.getContext() };

        auto kind{ op.getKind() };
        linalg::ElementwiseKind linalgKind;

        switch (kind) {
            case (dia::ElementwiseKind::square):
                linalgKind = linalg::ElementwiseKind::square;
                break;
            default:
                return failure();
        }

        auto inputType{ cast<RankedTensorType>(input.getType()) };
        int64_t rank{ inputType.getRank() };
        AffineMap identityMap{ rewriter.getMultiDimIdentityMap(rank) };

        auto indexingMapsAttr{ rewriter.getAffineMapArrayAttr({ identityMap, identityMap }) };

        SmallVector<Attribute> iteratorTypes(
            rank, linalg::IteratorTypeAttr::get(context, utils::IteratorType::parallel));
        auto iteratorTypesAttr{ rewriter.getArrayAttr(iteratorTypes) };

        SmallVector<NamedAttribute> attrs{
            rewriter.getNamedAttr("kind", linalg::ElementwiseKindAttr::get(context, linalgKind)),
            rewriter.getNamedAttr("indexing_maps", indexingMapsAttr),
            rewriter.getNamedAttr("iterator_types", iteratorTypesAttr)
        };

        auto kindAttr{ linalg::ElementwiseKindAttr::get(context, linalgKind) };
        auto namedKindAttr{ rewriter.getNamedAttr("kind", kindAttr) };

        auto elementwiseOp{ linalg::ElementwiseOp::create(rewriter, loc, ValueRange{ input },
                                                          ValueRange{ output }, attrs) };
        rewriter.replaceOp(op, elementwiseOp->getResults());
        return success();
    }

    LogicalResult diaTimesDenseToDenseBandedElementwiseToSCF(dia::ElementwiseOp op, PatternRewriter& rewriter,
                                                             const BandedSubMatrix& bandResult) const {
        Location loc{ op.getLoc() };
        Value A{ op.getInputs()[0] };
        Value B{ op.getInputs()[1] };
        Value C{ op.getOutput() };

        auto outputType{ cast<RankedTensorType>(C.getType()) };
        auto elementType{ cast<RankedTensorType>(A.getType()).getElementType() };
        MLIRContext* context{ rewriter.getContext() };

        BandedSubMatrix bandA{ BandedStructureAnalysis::readPropertyFromDictAttr(A.getDefiningOp()->getAttrDictionary()) };

        auto lA{ bandA.Property.LowerBandwidth };
        auto uA{ bandA.Property.UpperBandwidth };
        auto lC{ bandResult.Property.LowerBandwidth };
        auto uC{ bandResult.Property.UpperBandwidth };

        Value zero{ arith::ConstantOp::create(rewriter, loc, elementType, rewriter.getZeroAttr(elementType)) };
        Value zeroedC{ linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0) };

        auto c0{ arith::ConstantIndexOp::create(rewriter, loc, 0) };
        auto c1{ arith::ConstantIndexOp::create(rewriter, loc, 1) };
        auto cf0{ arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(elementType, 0.0)) };

        auto rank{ outputType.getRank() };
        auto totalRows{ arith::ConstantIndexOp::create(rewriter, loc, outputType.getDimSize(rank - 2)) };
        auto totalCols{ arith::ConstantIndexOp::create(rewriter, loc, outputType.getDimSize(rank - 1)) };

        auto cLA{ arith::ConstantIndexOp::create(rewriter, loc, lA) };
        auto currC{ zeroedC };

        if (lC > lA) {
            auto lDiff{ arith::ConstantIndexOp::create(rewriter, loc, lC - lA) };
            auto cLC{ arith::ConstantIndexOp::create(rewriter, loc, lC) };
            scf::ForOp iLoop{ scf::ForOp::create(
                rewriter, loc, c0, lDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::SubIOp::create(ob, loc, cLC, i).getResult() };
                    Value mMinusBand{ arith::SubIOp::create(ob, loc, totalRows, currBand).getResult() };
                    Value numEls{ arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult() };
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cRow{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            Value operand1{ cf0 };
                            Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B, ValueRange{ cRow, r }).getResult() };
                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                default:
                                    assert(false);
                            }
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner, ValueRange{ cRow, r }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        })};
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                })};
            currC = iLoop.getResult(0);
        }

        auto lMinVal{ std::min(lA, lC) };
        auto totalLowerDiags{ arith::ConstantIndexOp::create(rewriter, loc, lMinVal + 1) }; // do main diag
        auto lMin{ arith::ConstantIndexOp::create(rewriter, loc, lMinVal) };
        scf::ForOp iLoopLower{ scf::ForOp::create(
            rewriter, loc, c0, totalLowerDiags, c1, ValueRange{ currC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                auto cOuter{ iArgs[0] };
                Value currBand{ arith::SubIOp::create(ob, loc, lMin, i).getResult() };
                Value aInd{ arith::SubIOp::create(ob, loc, cLA, currBand).getResult() };
                Value mMinusBand{ arith::SubIOp::create(ob, loc, totalRows, currBand).getResult() };
                Value numEls{ arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult() };
                auto rLoop{ scf::ForOp::create(
                    ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                    [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                        auto cInner{ rArgs[0] };
                        Value cRow{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                        Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A, ValueRange{ aInd, r }).getResult() };
                        Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B, ValueRange{ cRow, r }).getResult() };
                        Value newOp;
                        switch (op.getKind()) {
                            case dia::ElementwiseKind::add:
                                newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            case dia::ElementwiseKind::sub:
                                newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            default:
                                assert(false);
                        }
                        auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner, ValueRange{ cRow, r }) };
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    })};
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            })};
        currC = iLoopLower.getResult(0);

        if (uC > uA) {
            auto uDiff{ arith::ConstantIndexOp::create(rewriter, loc, uC - uA) };
            auto startUpperBand{ arith::ConstantIndexOp::create(rewriter, loc, uA + 1) };
            scf::ForOp iLoop{ scf::ForOp::create(
                rewriter, loc, c0, uDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::AddIOp::create(ob, loc, i, startUpperBand).getResult() };
                    Value nMinusBand{ arith::SubIOp::create(ob, loc, totalCols, currBand).getResult() };
                    Value numEls{ arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult() };
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cCol{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            Value operand1{ cf0 };
                            Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B, ValueRange{ r, cCol }).getResult() };
                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                default: assert(false);
                            }
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner, ValueRange{ r, cCol }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        })};
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                })};
            currC = iLoop.getResult(0);
        }

        auto uMinVal{ std::min(uA, uC) };
        if (uMinVal != 0) {
            auto totalUpperDiags{ arith::ConstantIndexOp::create(rewriter, loc, uMinVal) };
            scf::ForOp iLoopUpper{ scf::ForOp::create(
                rewriter, loc, c0, totalUpperDiags, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::AddIOp::create(ob, loc, i, c1).getResult() };
                    Value aInd{ arith::AddIOp::create(ob, loc, cLA, currBand).getResult() };
                    Value nMinusBand{ arith::SubIOp::create(ob, loc, totalCols, currBand).getResult() };
                    Value numEls{ arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult() };
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cCol{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A, ValueRange{ aInd, r }).getResult() };
                            Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B, ValueRange{ r, cCol }).getResult() };
                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                default: assert(false);
                            }
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner, ValueRange{ r, cCol }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        })};
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                })};
            currC = iLoopUpper.getResult(0);
        }

        currC.getDefiningOp()->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, currC);
        return success();
    }


    LogicalResult denseTimesDiaToDenseBandedElementwiseToSCF(dia::ElementwiseOp op, PatternRewriter& rewriter,
                                                             const BandedSubMatrix& bandResult) const {
        Location loc{ op.getLoc() };
        Value A{ op.getInputs()[0] };
        Value B{ op.getInputs()[1] };
        Value C{ op.getOutput() };

        auto outputType{ cast<RankedTensorType>(C.getType()) };
        auto elementType{ cast<RankedTensorType>(A.getType()).getElementType() };
        MLIRContext* context{ rewriter.getContext() };

        BandedSubMatrix bandB{ BandedStructureAnalysis::readPropertyFromDictAttr(B.getDefiningOp()->getAttrDictionary()) };

        auto lB{ bandB.Property.LowerBandwidth };
        auto uB{ bandB.Property.UpperBandwidth };
        auto lC{ bandResult.Property.LowerBandwidth };
        auto uC{ bandResult.Property.UpperBandwidth };

        Value zero{ arith::ConstantOp::create(rewriter, loc, elementType, rewriter.getZeroAttr(elementType)) };
        Value zeroedC{ linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0) };

        auto c0{ arith::ConstantIndexOp::create(rewriter, loc, 0) };
        auto c1{ arith::ConstantIndexOp::create(rewriter, loc, 1) };
        auto cf0{ arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(elementType, 0.0)) };

        auto rank{ outputType.getRank() };
        auto totalRows{ arith::ConstantIndexOp::create(rewriter, loc, outputType.getDimSize(rank - 2)) };
        auto totalCols{ arith::ConstantIndexOp::create(rewriter, loc, outputType.getDimSize(rank - 1)) };

        auto cLB{ arith::ConstantIndexOp::create(rewriter, loc, lB) };
        auto currC{ zeroedC };

        if (lC > lB) {
            auto lDiff{ arith::ConstantIndexOp::create(rewriter, loc, lC - lB) };
            auto cLC{ arith::ConstantIndexOp::create(rewriter, loc, lC) };
            scf::ForOp iLoop{ scf::ForOp::create(
                rewriter, loc, c0, lDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::SubIOp::create(ob, loc, cLC, i).getResult() };
                    Value mMinusBand{ arith::SubIOp::create(ob, loc, totalRows, currBand).getResult() };
                    Value numEls{ arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult() };
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cRow{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A, ValueRange{ cRow, r }).getResult() };
                            Value operand2{ cf0 };
                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                default:
                                    assert(false);
                            }
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner, ValueRange{ cRow, r }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        })};
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                })};
            currC = iLoop.getResult(0);
        }

        auto lMinVal{ std::min(lB, lC) };
        auto totalLowerDiags{ arith::ConstantIndexOp::create(rewriter, loc, lMinVal + 1) };
        auto lMin{ arith::ConstantIndexOp::create(rewriter, loc, lMinVal) };
        scf::ForOp iLoopLower{ scf::ForOp::create(
            rewriter, loc, c0, totalLowerDiags, c1, ValueRange{ currC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                auto cOuter{ iArgs[0] };
                Value currBand{ arith::SubIOp::create(ob, loc, lMin, i).getResult() };
                Value bInd{ arith::SubIOp::create(ob, loc, cLB, currBand).getResult() };
                Value mMinusBand{ arith::SubIOp::create(ob, loc, totalRows, currBand).getResult() };
                Value numEls{ arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult() };
                auto rLoop{ scf::ForOp::create(
                    ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                    [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                        auto cInner{ rArgs[0] };
                        Value cRow{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                        Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A, ValueRange{ cRow, r }).getResult() };
                        Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B, ValueRange{ bInd, r }).getResult() };
                        Value newOp;
                        switch (op.getKind()) {
                            case dia::ElementwiseKind::add:
                                newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            case dia::ElementwiseKind::sub:
                                newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            default:
                                assert(false);
                        }
                        auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner, ValueRange{ cRow, r }) };
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    })};
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            })};
        currC = iLoopLower.getResult(0);

        if (uC > uB) {
            auto uDiff{ arith::ConstantIndexOp::create(rewriter, loc, uC - uB) };
            auto startUpperBand{ arith::ConstantIndexOp::create(rewriter, loc, uB + 1) };
            scf::ForOp iLoop{ scf::ForOp::create(
                rewriter, loc, c0, uDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::AddIOp::create(ob, loc, i, startUpperBand).getResult() };
                    Value nMinusBand{ arith::SubIOp::create(ob, loc, totalCols, currBand).getResult() };
                    Value numEls{ arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult() };
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cCol{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A, ValueRange{ r, cCol }).getResult() };
                            Value operand2{ cf0 };
                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                default:
                                    assert(false);
                            }
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner, ValueRange{ r, cCol }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        })};
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                })};
            currC = iLoop.getResult(0);
        }

        auto uMinVal{ std::min(uB, uC) };
        if (uMinVal != 0) {
            auto totalUpperDiags{ arith::ConstantIndexOp::create(rewriter, loc, uMinVal) };
            scf::ForOp iLoopUpper{ scf::ForOp::create(
                rewriter, loc, c0, totalUpperDiags, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::AddIOp::create(ob, loc, i, c1).getResult() };
                    Value bInd{ arith::AddIOp::create(ob, loc, cLB, currBand).getResult() };
                    Value nMinusBand{ arith::SubIOp::create(ob, loc, totalCols, currBand).getResult() };
                    Value numEls{ arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult() };
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cCol{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A, ValueRange{ r, cCol }).getResult() };
                            Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B, ValueRange{ bInd, r }).getResult() };
                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                default:
                                    assert(false);
                            }
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner, ValueRange{ r, cCol }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        })};
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                })};
            currC = iLoopUpper.getResult(0);
        }

        currC.getDefiningOp()->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, currC);
        return success();
    }

    LogicalResult matchAndRewrite(dia::ElementwiseOp op, PatternRewriter& rewriter) const override {
        auto dict = op->getAttrDictionary();
        if (!dict) dict = DictionaryAttr();

        const BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        auto lower = opBandInfo.Property.LowerBandwidth;
        auto upper = opBandInfo.Property.UpperBandwidth;

        BandedSubMatrix bandA;
        BandedSubMatrix bandB;

        auto numOps{ op.getInputs().size() };

        if (numOps == 2) {
            Value A = op.getInputs()[0];
            Value B = op.getInputs()[1];
            Value C = op.getOutput();

            Operation* defOpA = A.getDefiningOp();
            Operation* defOpB = B.getDefiningOp();

            auto dictA = defOpA->getAttrDictionary();
            auto dictB = defOpB->getAttrDictionary();

            if (!dictA || !dictB) return failure();

            bandA = BandedStructureAnalysis::readPropertyFromDictAttr(dictA);
            bandB = BandedStructureAnalysis::readPropertyFromDictAttr(dictB);

            // diagonal possible combinations
            // if (opBandInfo.isDiagonal()) {
            //     return failure();
            // } else {
            if (bandA.IsDia && bandB.IsDia && opBandInfo.IsDia)
                return diaTimesDiaToDiaBandedElementwiseToSCF(op, rewriter, opBandInfo);
            else if (bandA.IsDia && bandB.IsDia && !opBandInfo.IsDia)
                return diaTimesDiaToDenseBandedElementwiseToSCF(op, rewriter, opBandInfo);
            else if (bandA.IsDia && !bandB.IsDia && !opBandInfo.IsDia)
                return diaTimesDenseToDenseBandedElementwiseToSCF(op, rewriter, opBandInfo);
            else if (bandA.IsDia && !bandB.IsDia && opBandInfo.IsDia)
                return diaTimesDenseToDiaBandedElementwiseToSCF(op, rewriter, opBandInfo);
            else if (!bandA.IsDia && bandB.IsDia && !opBandInfo.IsDia)
                return denseTimesDiaToDenseBandedElementwiseToSCF(op, rewriter, opBandInfo);
            else if (!bandA.IsDia && bandB.IsDia && opBandInfo.IsDia)
                return denseTimesDiaToDiaBandedElementwiseToSCF(op, rewriter, opBandInfo);
            else
                return failure();
            // }
        } else if (numOps == 1) {
            Value A = op.getInputs()[0];
            Value C = op.getOutput();

            Operation* defOpA = A.getDefiningOp();

            auto dictA = defOpA->getAttrDictionary();

            if (!dictA) return failure();
            bandA = BandedStructureAnalysis::readPropertyFromDictAttr(dictA);

            // diagonal possible combinations
            if (opBandInfo.isDiagonal()) {
                return failure();
            } else
                return diaToDiaBandedElementwiseToLinalg(op, rewriter, opBandInfo);
        } else
            return failure();
    }
};

// ------------------------------------------------------------------------------------------------------------------------------
// dia.MatmulOp
// ------------------------------------------------------------------------------------------------------------------------------

struct DIAMatMulPattern : public OpRewritePattern<dia::MatmulOp> {
    using OpRewritePattern::OpRewritePattern;
    DIAMatMulPattern(MLIRContext* ctx, bool detectDIA)
        : OpRewritePattern(ctx), detectDIA(detectDIA) {
    }

    LogicalResult diaTimesDenseToDiaDiagMatmulToLinalg(dia::MatmulOp op,
                                                       PatternRewriter& rewriter) const {
        Location loc = op.getLoc();
        Value A = op.getLhs();
        Value B = op.getRhs();
        Value C = op.getOutput();

        MLIRContext* context = rewriter.getContext();

        AffineExpr d0 = rewriter.getAffineDimExpr(0);
        AffineExpr zero = rewriter.getAffineConstantExpr(0);

        AffineMap diaInMap = AffineMap::get(1, 0, { zero, d0 }, context);
        AffineMap denseMap = AffineMap::get(1, 0, { d0, d0 }, context);
        AffineMap diaOutMap = AffineMap::get(1, 0, { zero, d0 }, context);

        SmallVector<AffineMap, 3> indexingMaps = {
            diaInMap,
            denseMap,
            diaOutMap,
        };

        llvm::SmallVector<utils::IteratorType, 1> iteratorTypes = { utils::IteratorType::parallel };

        auto genericOp = linalg::GenericOp::create(
            rewriter, loc, TypeRange{ op.getOutput().getType() }, ValueRange{ A, B },
            ValueRange{ C }, indexingMaps, iteratorTypes,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                Value mul = arith::MulFOp::create(b, loc, args[0], args[1]);
                linalg::YieldOp::create(b, loc, ValueRange{ mul });
            });
        genericOp->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, genericOp);
        return success();
    }

    // dia diag should always result in dia
    LogicalResult diaTimesDiaToDiaDiagMatmulToLinalg(dia::MatmulOp op,
                                                     PatternRewriter& rewriter) const {
        Location loc = op.getLoc();
        Value A = op.getLhs();
        Value B = op.getRhs();
        Value C = op.getOutput();

        MLIRContext* context = rewriter.getContext();

        AffineExpr d0 = rewriter.getAffineDimExpr(0);

        SmallVector<AffineMap, 3> indexingMaps = {
            AffineMap::getMultiDimIdentityMap(2, context),
            AffineMap::getMultiDimIdentityMap(2, context),
            AffineMap::getMultiDimIdentityMap(2, context),
        };

        llvm::SmallVector<utils::IteratorType, 2> iteratorTypes = { utils::IteratorType::parallel,
                                                                    utils::IteratorType::parallel };

        auto genericOp = linalg::GenericOp::create(
            rewriter, loc, TypeRange{ op.getOutput().getType() }, ValueRange{ A, B },
            ValueRange{ C }, indexingMaps, iteratorTypes,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                Value mul = arith::MulFOp::create(b, loc, args[0], args[1]);
                linalg::YieldOp::create(b, loc, ValueRange{ mul });
            });
        genericOp->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, genericOp);
        return success();
    }

    LogicalResult diaTimesDiaToDiaBandedMatmulToSCF(dia::MatmulOp op, PatternRewriter& rewriter,
                                                    const BandedSubMatrix& bandResult) const {
        Location loc = op.getLoc();
        Value A = op.getLhs();
        Value B = op.getRhs();
        Value C = op.getOutput();

        Operation* defOpA = A.getDefiningOp();
        Operation* defOpB = B.getDefiningOp();

        auto dictA = defOpA->getAttrDictionary();
        auto dictB = defOpB->getAttrDictionary();

        if (!dictA || !dictB) return failure();

        const BandedSubMatrix bandA = BandedStructureAnalysis::readPropertyFromDictAttr(dictA);
        const BandedSubMatrix bandB = BandedStructureAnalysis::readPropertyFromDictAttr(dictB);

        auto resultType = cast<RankedTensorType>(A.getType());
        const uint64_t N = resultType.getDimSize(1);

        const int64_t lA = bandA.Property.LowerBandwidth;
        const int64_t uA = bandA.Property.UpperBandwidth;
        const int64_t lB = bandB.Property.LowerBandwidth;
        const int64_t uB = bandB.Property.UpperBandwidth;
        const int64_t lC = bandResult.Property.LowerBandwidth;
        const int64_t uC = bandResult.Property.UpperBandwidth;

        auto elementType = cast<RankedTensorType>(A.getType()).getElementType();

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value dimN = arith::ConstantIndexOp::create(rewriter, loc, N);

        Value lAVal = arith::ConstantIndexOp::create(rewriter, loc, lA);
        Value uAVal = arith::ConstantIndexOp::create(rewriter, loc, uA);
        Value lBVal = arith::ConstantIndexOp::create(rewriter, loc, lB);
        Value uBVal = arith::ConstantIndexOp::create(rewriter, loc, uB);
        Value lCVal = arith::ConstantIndexOp::create(rewriter, loc, lC);
        Value uCVal = arith::ConstantIndexOp::create(rewriter, loc, uC);

        // negative constants for lower bounds
        Value negLA = arith::ConstantIndexOp::create(rewriter, loc, -lA);
        Value negLB = arith::ConstantIndexOp::create(rewriter, loc, -lB);
        Value negLC = arith::ConstantIndexOp::create(rewriter, loc, -lC);

        Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                               rewriter.getZeroAttr(elementType));
        Value zeroedC =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0);

        // outer loop: for i in range(-lc, uc + 1)
        Value iStart = negLC;
        Value iEnd = arith::AddIOp::create(rewriter, loc, uCVal, c1);

        // outer loop: for i_shifted in range(0, lC + uC + 1)
        Value totalDiags = arith::ConstantIndexOp::create(rewriter, loc, lC + uC + 1);

        auto csti64 = [&](int64_t val) {
            return arith::ConstantIntOp::create(rewriter, loc, val, 64);
        };
        auto toIndex = [&](OpBuilder& b, Value v) {
            return arith::IndexCastOp::create(b, loc, rewriter.getIndexType(), v);
        };
        auto toI64 = [&](OpBuilder& b, Value v) {
            return arith::IndexCastOp::create(b, loc, rewriter.getIntegerType(64), v);
        };

        // constants as i64
        Value lAi64 = csti64(lA);
        Value uAi64 = csti64(uA);
        Value lBi64 = csti64(lB);
        Value uBi64 = csti64(uB);
        Value lCi64 = csti64(lC);
        Value Ni64 = csti64(N);
        Value c0i64 = csti64(0);
        Value c1i64 = csti64(1);

        // outer loop still uses index (0 to lC+uC+1)
        auto iLoop = scf::ForOp::create(
            rewriter, loc, c0, totalDiags, c1, ValueRange{ zeroedC },
            [&](OpBuilder& ob, Location loc, Value iShifted, ValueRange iArgs) {
                Value cOuter = iArgs[0];

                // i = iShifted - lC, computed in i64
                Value iShiftedI64 = toI64(ob, iShifted);
                Value i = arith::SubIOp::create(ob, loc, iShiftedI64, lCi64);

                // cIdx = iShifted
                Value cIdx = iShifted;

                // j_start = max(-lA, i - uB) in i64
                Value negLA = csti64(-lA);
                Value iMinusUb = arith::SubIOp::create(ob, loc, i, uBi64);
                Value jStartI64 = arith::MaxSIOp::create(ob, loc, negLA, iMinusUb);

                // j_end = min(uA, i + lB) + 1 in i64
                Value iPlusLb = arith::AddIOp::create(ob, loc, i, lBi64);
                Value jEndI64 = arith::AddIOp::create(
                    ob, loc, arith::MinSIOp::create(ob, loc, uAi64, iPlusLb), c1i64);

                // shift j to index: j_shifted_start = j_start + lA
                Value jStartShifted = toIndex(ob, arith::AddIOp::create(ob, loc, jStartI64, lAi64));
                Value jEndShifted = toIndex(ob, arith::AddIOp::create(ob, loc, jEndI64, lAi64));
                Value maxDiaIdx = arith::ConstantIndexOp::create(ob, loc, lA + uA + 1);
                Value jEndShiftedClamped = arith::MinSIOp::create(ob, loc, jEndShifted, maxDiaIdx);

                auto jLoop = scf::ForOp::create(
                    ob, loc, jStartShifted, jEndShiftedClamped, c1, ValueRange{ cOuter },
                    [&](OpBuilder& mb, Location loc, Value jShifted, ValueRange jArgs) {
                        Value cMid = jArgs[0];

                        // j = jShifted - lA in i64
                        Value jShiftedI64 = toI64(mb, jShifted);
                        Value j = arith::SubIOp::create(mb, loc, jShiftedI64, lAi64);

                        Value aIdx = jShifted;

                        // k = i - j in i64, bIdx = k + lB
                        Value k = arith::SubIOp::create(mb, loc, i, j);
                        Value bIdx = toIndex(mb, arith::AddIOp::create(mb, loc, k, lBi64));
                        Value maxBDiaIdx = arith::ConstantIndexOp::create(mb, loc, lB + uB);
                        Value bIdxClamped = arith::MinSIOp::create(mb, loc, bIdx, maxBDiaIdx);

                        // r_start = max(0, -j) in i64
                        Value negJ = arith::SubIOp::create(mb, loc, c0i64, j);
                        Value rStart = toIndex(mb, arith::MaxSIOp::create(mb, loc, c0i64, negJ));

                        // r_end = min(N, N - j) in i64
                        Value nMinusJ = arith::SubIOp::create(mb, loc, Ni64, j);
                        Value rEnd = toIndex(mb, arith::MinSIOp::create(mb, loc, Ni64, nMinusJ));

                        auto rLoop = scf::ForOp::create(
                            mb, loc, rStart, rEnd, c1, ValueRange{ cMid },
                            [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                                Value cInner = rArgs[0];

                                Value cVal = tensor::ExtractOp::create(ib, loc, cInner,
                                                                       ValueRange{ cIdx, r });
                                Value aVal =
                                    tensor::ExtractOp::create(ib, loc, A, ValueRange{ aIdx, r });

                                // r + j in i64, then cast to index
                                Value rI64 = toI64(ib, r);
                                Value rPlusJ = toIndex(ib, arith::AddIOp::create(ib, loc, rI64, j));
                                Value bVal = tensor::ExtractOp::create(
                                    ib, loc, B, ValueRange{ bIdxClamped, rPlusJ });

                                Value mul = arith::MulFOp::create(ib, loc, aVal, bVal);
                                Value acc = arith::AddFOp::create(ib, loc, cVal, mul);
                                Value updated = tensor::InsertOp::create(ib, loc, acc, cInner,
                                                                         ValueRange{ cIdx, r });

                                scf::YieldOp::create(ib, loc, ValueRange{ updated });
                            });

                        scf::YieldOp::create(mb, loc, rLoop.getResults());
                    });

                scf::YieldOp::create(ob, loc, jLoop.getResults());
            });
        iLoop->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, iLoop.getResult(0));
        return success();
    }

    LogicalResult diaTimesDiaToDenseBandedMatmulToSCF(dia::MatmulOp op,
                                                      PatternRewriter& rewriter) const {
        Location loc = op->getLoc();
        Value A = op.getLhs();
        Value B = op.getRhs();
        Value C = op.getOutput();

        auto resultType = cast<RankedTensorType>(C.getType());
        const int64_t N = resultType.getDimSize(1);

        Operation* defOpA = A.getDefiningOp();
        Operation* defOpB = B.getDefiningOp();

        auto dictA = defOpA->getAttrDictionary();
        auto dictB = defOpB->getAttrDictionary();

        if (!dictA || !dictB) return failure();

        const BandedSubMatrix bandA = BandedStructureAnalysis::readPropertyFromDictAttr(dictA);
        const BandedSubMatrix bandB = BandedStructureAnalysis::readPropertyFromDictAttr(dictB);
        const uint64_t upperA = bandA.Property.UpperBandwidth;
        const uint64_t lowerA = bandA.Property.LowerBandwidth;

        const uint64_t upperB = bandB.Property.UpperBandwidth;
        const uint64_t lowerB = bandB.Property.LowerBandwidth;

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value cN = arith::ConstantIndexOp::create(rewriter, loc, N);
        Value cLA = arith::ConstantIndexOp::create(rewriter, loc, lowerA);
        Value cUA = arith::ConstantIndexOp::create(rewriter, loc, upperA);
        Value cLB = arith::ConstantIndexOp::create(rewriter, loc, lowerB);
        Value cUB = arith::ConstantIndexOp::create(rewriter, loc, upperB);
        Value cLAi64 = arith::ConstantIntOp::create(rewriter, loc, lowerA, 64);
        Value cLBi64 = arith::ConstantIntOp::create(rewriter, loc, lowerB, 64);

        auto elementType = resultType.getElementType();
        Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                               rewriter.getZeroAttr(elementType));
        Value zeroedC =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0);

        auto toIndex = [&](OpBuilder& b, Value v) {
            return arith::IndexCastOp::create(b, loc, b.getIndexType(), v);
        };
        auto toI64 = [&](OpBuilder& b, Value v) {
            return arith::IndexCastOp::create(b, loc, b.getI64Type(), v);
        };

        auto iLoop = scf::ForOp::create(
            rewriter, loc, c0, cN, c1, ValueRange{ zeroedC },
            [&](OpBuilder& ob, Location loc, Value row, ValueRange rowArgs) {
                Value cRow = rowArgs[0];

                auto jLoop = scf::ForOp::create(
                    ob, loc, c0, cN, c1, ValueRange{ cRow },
                    [&](OpBuilder& cb, Location loc, Value col, ValueRange colArgs) {
                        Value cCol = colArgs[0];

                        // kstart = max(0, row - lA, col - uB)
                        Value rowMinusLA = arith::SubIOp::create(cb, loc, row, cLA);
                        Value colMinusUB = arith::SubIOp::create(cb, loc, col, cUB);
                        Value kstart0 = arith::MaxSIOp::create(cb, loc, c0, rowMinusLA);
                        Value kstart = arith::MaxSIOp::create(cb, loc, kstart0, colMinusUB);

                        // kend = min(N, row + uA + 1, col + lB + 1)
                        Value rowPlusUA1 = arith::AddIOp::create(
                            cb, loc, arith::AddIOp::create(cb, loc, row, cUA), c1);
                        Value colPlusLB1 = arith::AddIOp::create(
                            cb, loc, arith::AddIOp::create(cb, loc, col, cLB), c1);
                        Value kend0 = arith::MinSIOp::create(cb, loc, cN, rowPlusUA1);
                        Value kend = arith::MinSIOp::create(cb, loc, kend0, colPlusLB1);

                        auto kLoop = scf::ForOp::create(
                            cb, loc, kstart, kend, c1, ValueRange{ cCol },
                            [&](OpBuilder& kb, Location loc, Value k, ValueRange kArgs) {
                                Value cK = kArgs[0];

                                // d_A = (k - row) + lA
                                Value rowI64 = toI64(kb, row);
                                Value kI64 = toI64(kb, k);
                                Value colI64 = toI64(kb, col);
                                Value dA = toIndex(
                                    kb, arith::AddIOp::create(
                                            kb, loc, arith::SubIOp::create(kb, loc, kI64, rowI64),
                                            cLAi64));

                                // d_B = (col - k) + lB
                                Value dB = toIndex(
                                    kb, arith::AddIOp::create(
                                            kb, loc, arith::SubIOp::create(kb, loc, colI64, kI64),
                                            cLBi64));

                                // A[d_A][row], B[d_B][k]
                                Value aVal =
                                    tensor::ExtractOp::create(kb, loc, A, ValueRange{ dA, row });
                                Value bVal =
                                    tensor::ExtractOp::create(kb, loc, B, ValueRange{ dB, k });

                                // C[row][col] += A[d_A][row] * B[d_B][k]
                                Value cVal =
                                    tensor::ExtractOp::create(kb, loc, cK, ValueRange{ row, col });
                                Value mul = arith::MulFOp::create(kb, loc, aVal, bVal);
                                Value acc = arith::AddFOp::create(kb, loc, cVal, mul);
                                Value updated = tensor::InsertOp::create(kb, loc, acc, cK,
                                                                         ValueRange{ row, col });

                                scf::YieldOp::create(kb, loc, ValueRange{ updated });
                            });

                        scf::YieldOp::create(cb, loc, kLoop.getResults());
                    });

                scf::YieldOp::create(ob, loc, jLoop.getResults());
            });

        rewriter.replaceOp(op, iLoop.getResult(0));
        return success();
    }

    LogicalResult diaTimesDenseTodiaBandedMatmulToSCF(dia::MatmulOp op,
                                                      PatternRewriter& rewriter) const {
        Location loc = op->getLoc();
        Value A = op.getLhs();
        Value B = op.getRhs();
        Value C = op.getOutput();

        auto resultType = cast<RankedTensorType>(C.getType());
        const int64_t N = resultType.getDimSize(1);

        Operation* defOpA = A.getDefiningOp();
        Operation* defOpB = B.getDefiningOp();

        auto dictA = defOpA->getAttrDictionary();
        auto dictB = defOpB->getAttrDictionary();

        if (!dictA || !dictB) return failure();

        const BandedSubMatrix bandA = BandedStructureAnalysis::readPropertyFromDictAttr(dictA);
        const BandedSubMatrix bandB = BandedStructureAnalysis::readPropertyFromDictAttr(dictB);
        const uint64_t upperA = bandA.Property.UpperBandwidth;
        const uint64_t lowerA = bandA.Property.LowerBandwidth;

        const uint64_t upperB = bandB.Property.UpperBandwidth;
        const uint64_t lowerB = bandB.Property.LowerBandwidth;

        int64_t olower = std::min(static_cast<uint64_t>(N - 1), (lowerA + lowerB));
        int64_t oupper = std::min(static_cast<uint64_t>(N - 1), upperA + upperB);
        int64_t C_rows = olower + oupper + 1;

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value cN = arith::ConstantIndexOp::create(rewriter, loc, N);
        Value cCRows = arith::ConstantIndexOp::create(rewriter, loc, C_rows);
        Value cOLower = arith::ConstantIndexOp::create(rewriter, loc, olower);
        Value cLA = arith::ConstantIndexOp::create(rewriter, loc, lowerA);
        Value cUA = arith::ConstantIndexOp::create(rewriter, loc, upperA);
        Value cLB = arith::ConstantIndexOp::create(rewriter, loc, lowerB);
        Value cUB = arith::ConstantIndexOp::create(rewriter, loc, upperB);
        Value cLAi64 = arith::ConstantIntOp::create(rewriter, loc, lowerA, 64);

        auto elementType = resultType.getElementType();
        Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                               rewriter.getZeroAttr(elementType));
        Value zeroedC =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0);

        auto toIndex = [&](OpBuilder& b, Value v) {
            return arith::IndexCastOp::create(b, loc, b.getIndexType(), v);
        };
        auto toI64 = [&](OpBuilder& b, Value v) {
            return arith::IndexCastOp::create(b, loc, b.getI64Type(), v);
        };

        // for d_C in range(C_rows)
        auto dLoop = scf::ForOp::create(
            rewriter, loc, c0, cCRows, c1, ValueRange{ zeroedC },
            [&](OpBuilder& db, Location loc, Value dC, ValueRange dArgs) {
                Value cOut = dArgs[0];

                // diag_offset = d_C - olower
                Value diagOffset = arith::SubIOp::create(db, loc, dC, cOLower);

                // for row in range(N)
                auto rowLoop = scf::ForOp::create(
                    db, loc, c0, cN, c1, ValueRange{ cOut },
                    [&](OpBuilder& rb, Location loc, Value row, ValueRange rowArgs) {
                        Value cRow = rowArgs[0];

                        // col = row + diag_offset
                        Value rowI64 = toI64(rb, row);
                        Value offsetI64 = toI64(rb, diagOffset);
                        Value colI64 = arith::AddIOp::create(rb, loc, rowI64, offsetI64);
                        Value col = toIndex(rb, colI64);

                        // if col < 0 or col >= N: skip
                        Value colValid0 =
                            arith::CmpIOp::create(rb, loc, arith::CmpIPredicate::sge, col, c0);
                        Value colValidN =
                            arith::CmpIOp::create(rb, loc, arith::CmpIPredicate::slt, col, cN);
                        Value colValid = arith::AndIOp::create(rb, loc, colValid0, colValidN);

                        auto ifOp =
                            scf::IfOp::create(rb, loc, TypeRange{ cRow.getType() }, colValid, true);

                        // then block
                        {
                            OpBuilder::InsertionGuard g(rb);
                            rb.setInsertionPointToStart(ifOp.thenBlock());

                            // kstart = max(0, row - lA, col - uB)
                            Value rowMinusLA = arith::SubIOp::create(rb, loc, row, cLA);
                            Value colMinusUB = arith::SubIOp::create(rb, loc, col, cUB);
                            Value ks0 = arith::MaxSIOp::create(rb, loc, c0, rowMinusLA);
                            Value kstart = arith::MaxSIOp::create(rb, loc, ks0, colMinusUB);

                            // kend = min(N, row + uA + 1, col + lB + 1)
                            Value rowPlusUA1 = arith::AddIOp::create(
                                rb, loc, arith::AddIOp::create(rb, loc, row, cUA), c1);
                            Value colPlusLB1 = arith::AddIOp::create(
                                rb, loc, arith::AddIOp::create(rb, loc, col, cLB), c1);
                            Value ke0 = arith::MinSIOp::create(rb, loc, cN, rowPlusUA1);
                            Value kend = arith::MinSIOp::create(rb, loc, ke0, colPlusLB1);

                            // for k in range(kstart, kend)
                            auto kLoop = scf::ForOp::create(
                                rb, loc, kstart, kend, c1, ValueRange{ cRow },
                                [&](OpBuilder& kb, Location loc, Value k, ValueRange kArgs) {
                                    Value cK = kArgs[0];

                                    // d_A = (k - row) + lA
                                    Value kI64 = toI64(kb, k);
                                    Value rI64 = toI64(kb, row);
                                    Value dA = toIndex(
                                        kb, arith::AddIOp::create(
                                                kb, loc, arith::SubIOp::create(kb, loc, kI64, rI64),
                                                cLAi64));

                                    // A[d_A][row], B[k][col]
                                    Value aVal = tensor::ExtractOp::create(kb, loc, A,
                                                                           ValueRange{ dA, row });
                                    Value bVal =
                                        tensor::ExtractOp::create(kb, loc, B, ValueRange{ k, col });

                                    // C[d_C][row] += A[d_A][row] * B[k][col]
                                    Value cVal = tensor::ExtractOp::create(kb, loc, cK,
                                                                           ValueRange{ dC, row });
                                    Value mul = arith::MulFOp::create(kb, loc, aVal, bVal);
                                    Value acc = arith::AddFOp::create(kb, loc, cVal, mul);
                                    Value updated = tensor::InsertOp::create(kb, loc, acc, cK,
                                                                             ValueRange{ dC, row });

                                    scf::YieldOp::create(kb, loc, ValueRange{ updated });
                                });

                            scf::YieldOp::create(rb, loc, kLoop.getResults());
                        }

                        // else block — yield unchanged tensor
                        {
                            OpBuilder::InsertionGuard g(rb);
                            rb.setInsertionPointToStart(ifOp.elseBlock());
                            scf::YieldOp::create(rb, loc, ValueRange{ cRow });
                        }

                        scf::YieldOp::create(rb, loc, ifOp.getResults());
                    });

                scf::YieldOp::create(db, loc, rowLoop.getResults());
            });

        rewriter.replaceOp(op, dLoop.getResult(0));
        return success();
    }
    /// Rewrite a dense `dia.matmul` into an explicit SCF loop nest that
    /// computes only the entries within the intersection.
    /// The result is materialized in the DIA format.
    LogicalResult denseTimesDenseToDiaBandedMatmulToSCF(dia::MatmulOp op, PatternRewriter& rewriter,
                                                        const BandedSubMatrix& outputBand) const {
        Location loc = op.getLoc();
        Value A = op.getLhs();
        Value B = op.getRhs();
        Value C = op.getOutput();

        Operation* defOpA = A.getDefiningOp();
        Operation* defOpB = B.getDefiningOp();

        auto dictA = defOpA->getAttrDictionary();
        auto dictB = defOpB->getAttrDictionary();

        if (!dictA || !dictB) return failure();

        const BandedSubMatrix bandA = BandedStructureAnalysis::readPropertyFromDictAttr(dictA);
        const BandedSubMatrix bandB = BandedStructureAnalysis::readPropertyFromDictAttr(dictB);

        auto resultType = cast<RankedTensorType>(op.getResult().getType());
        auto inputType = cast<RankedTensorType>(A.getType());

        const uint64_t N = inputType.getDimSize(0);
        const uint64_t M = inputType.getDimSize(1);

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
                        Value kStart0 = arith::MaxSIOp::create(mb, loc, jMinusUb, iMinusLa);
                        Value kStart = arith::MaxSIOp::create(mb, loc, c0, kStart0);

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
        iLoop->setAttr("metadata", op->getAttrOfType<DictionaryAttr>("metadata"));
        rewriter.replaceOp(op, iLoop.getResult(0));
        return success();
    }

    LogicalResult denseTimesDenseToDiaDiagMatmulToLinalg(dia::MatmulOp op,
                                                         PatternRewriter& rewriter) const {
        auto loc = op.getLoc();
        Value A = op.getLhs();
        Value B = op.getRhs();
        Value C = op.getOutput();
        MLIRContext* context = rewriter.getContext();
        auto matrixType = cast<RankedTensorType>(C.getType());
        int64_t M = matrixType.getDimSize(1);
        auto elementType = matrixType.getElementType();

        auto vectorType = RankedTensorType::get({ 1, M }, elementType);
        Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                               rewriter.getZeroAttr(elementType));
        Value emptyVec = tensor::EmptyOp::create(rewriter, loc, vectorType, ValueRange{});
        Value zeroedVec =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ emptyVec })
                .getResult(0);

        AffineExpr d0 = rewriter.getAffineDimExpr(0);
        AffineExpr c0 = rewriter.getAffineConstantExpr(0);

        // A[d0][d0], B[d0][d0] -> C[0][d0]
        AffineMap diagMap = AffineMap::get(1, 0, { d0, d0 }, context);
        AffineMap outputMap = AffineMap::get(1, 0, { c0, d0 }, context);

        SmallVector<AffineMap, 3> indexingMaps = { diagMap, diagMap, outputMap };
        SmallVector<utils::IteratorType, 1> iteratorTypes = { utils::IteratorType::parallel };

        auto genericOp = linalg::GenericOp::create(
            rewriter, loc, TypeRange{ vectorType }, ValueRange{ A, B }, ValueRange{ zeroedVec },
            indexingMaps, iteratorTypes, [&](OpBuilder& b, Location loc, ValueRange args) {
                Value mul = arith::MulFOp::create(b, loc, args[0], args[1]);
                linalg::YieldOp::create(b, loc, ValueRange{ mul });
            });

        if (auto metadata = op->getAttr("metadata")) genericOp->setAttr("metadata", metadata);

        rewriter.replaceOp(op, genericOp);
        return success();
    }

    LogicalResult matchAndRewrite(dia::MatmulOp op, PatternRewriter& rewriter) const override {
        auto dict = op->getAttrDictionary();
        if (!dict) dict = DictionaryAttr();

        const BandedSubMatrix resultBand = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        auto lower = resultBand.Property.LowerBandwidth;
        auto upper = resultBand.Property.UpperBandwidth;

        Value A = op.getLhs();
        Value B = op.getRhs();
        Value C = op.getOutput();

        Operation* defOpA = A.getDefiningOp();
        Operation* defOpB = B.getDefiningOp();

        auto dictA = defOpA->getAttrDictionary();
        auto dictB = defOpB->getAttrDictionary();

        if (!dictA || !dictB) return failure();

        const BandedSubMatrix bandA = BandedStructureAnalysis::readPropertyFromDictAttr(dictA);
        const BandedSubMatrix bandB = BandedStructureAnalysis::readPropertyFromDictAttr(dictB);

        // diagonal possible combinations
        if (resultBand.isDiagonal()) {
            if (!bandA.IsDia && !bandB.IsDia)
                return denseTimesDenseToDiaDiagMatmulToLinalg(op, rewriter);
            else if (bandA.IsDia && !bandB.IsDia)
                return diaTimesDenseToDiaDiagMatmulToLinalg(op, rewriter);
            else if (!bandA.IsDia && bandB.IsDia)
                return failure();
            // TODO: implement function below
            //  return denseTimesDiaToDiaDiagMatmulToLinalg(op, rewriter);
            else
                return diaTimesDiaToDiaDiagMatmulToLinalg(op, rewriter);
        } else {
            // Output should be mapped to dense when:
            // inputs are in DIA, the analysis concluded that the result is not DIA
            // the `detect-dia` flag is `true`
            if (bandA.IsDia && bandB.IsDia && !resultBand.IsDia && detectDIA) {
                return diaTimesDiaToDenseBandedMatmulToSCF(op, rewriter);
            } else if (bandA.IsDia && !bandB.IsDia && resultBand.IsDia) {
                return diaTimesDenseTodiaBandedMatmulToSCF(op, rewriter);
            } else if (!bandA.IsDia && !bandB.IsDia && resultBand.IsDia) {
                return denseTimesDenseToDiaBandedMatmulToSCF(op, rewriter, resultBand);
            }

            return diaTimesDiaToDiaBandedMatmulToSCF(op, rewriter, resultBand);
        }
    }

   private:
    bool detectDIA;
};

// ------------------------------------------------------------------------------------------------------------------------------
// linalg.BatchMatmul
// ------------------------------------------------------------------------------------------------------------------------------

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
                Value cOut = bArgs[0];

                auto iLoop = scf::ForOp::create(
                    bb, loc, c0, dimN, c1, ValueRange{ cOut },
                    [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                        Value cMid = iArgs[0];

                        // k start = max(0, i - (lA + lB))
                        Value lAplB = arith::AddIOp::create(ob, loc, lowerA, lowerB);
                        Value iMinusLALB = arith::SubIOp::create(ob, loc, i, lAplB);
                        Value kStart = arith::MaxSIOp::create(ob, loc, c0, iMinusLALB);

                        // k end = min(M, i + (uA + uB) + 1)
                        Value uApuB = arith::AddIOp::create(ob, loc, upperA, upperB);
                        Value iPlusuAuB = arith::AddIOp::create(ob, loc, i, uApuB);
                        Value kEndRaw = arith::AddIOp::create(ob, loc, iPlusuAuB, c1);
                        Value kEnd = arith::MinSIOp::create(ob, loc, dimM, kEndRaw);

                        auto kLoop = scf::ForOp::create(
                            ob, loc, kStart, kEnd, c1, ValueRange{ cMid },
                            [&](OpBuilder& mb, Location loc, Value k, ValueRange kArgs) {
                                Value cInner = kArgs[0];

                                // j start = max(0, max(i - lA, k - uB))
                                Value iMinusLa = arith::SubIOp::create(mb, loc, i, lowerA);
                                Value kMinusUb = arith::SubIOp::create(mb, loc, k, upperB);
                                Value jStartInner =
                                    arith::MaxSIOp::create(mb, loc, iMinusLa, kMinusUb);
                                Value jStart = arith::MaxSIOp::create(mb, loc, c0, jStartInner);

                                // j end = min(N, min(i + uA, k + lB) + 1)
                                Value iPlusUa = arith::AddIOp::create(mb, loc, i, upperA);
                                Value kPlusLb = arith::AddIOp::create(mb, loc, k, lowerB);
                                Value jEndInner = arith::MinSIOp::create(mb, loc, iPlusUa, kPlusLb);
                                Value jEndRaw = arith::AddIOp::create(mb, loc, jEndInner, c1);
                                Value jEnd = arith::MinSIOp::create(mb, loc, dimN, jEndRaw);

                                auto jLoop = scf::ForOp::create(
                                    mb, loc, jStart, jEnd, c1, ValueRange{ cInner },
                                    [&](OpBuilder& ib, Location loc, Value j, ValueRange jArgs) {
                                        Value cInnest = jArgs[0];
                                        Value cbik = tensor::ExtractOp::create(
                                            ib, loc, cInnest, ValueRange{ b, i, k });
                                        Value abij = tensor::ExtractOp::create(
                                            ib, loc, A, ValueRange{ b, i, j });
                                        Value bbjk = tensor::ExtractOp::create(
                                            ib, loc, B, ValueRange{ b, j, k });
                                        Value mul = arith::MulFOp::create(ib, loc, abij, bbjk);
                                        Value add = arith::AddFOp::create(ib, loc, cbik, mul);
                                        Value updated = tensor::InsertOp::create(
                                            ib, loc, add, cInnest, ValueRange{ b, i, k });
                                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                                    });
                                scf::YieldOp::create(mb, loc, jLoop.getResults());
                            });
                        scf::YieldOp::create(ob, loc, kLoop.getResults());
                    });
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

        if (opBandInfo.isDiagonal()) return denseTimesDenseToDenseDiagMatmulToLinalg(op, rewriter);
        // banded
        else
            return denseTimesDenseToDenseBandedMatmulToSCF(op, rewriter);
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
                        Value jStart = arith::MaxSIOp::create(
                            mb, loc, arith::MaxSIOp::create(mb, loc, kMinusUb, iMinusLa), c0);

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

        auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
        uint64_t n = resultType.getDimSize(0);
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
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = funcOp.getContext();

        // check if the analysis propagated dia flags
        auto cached = getCachedAnalysis<BandedAnalysisResult>();
        bool detectDIA = cached ? cached->get().detectDIA : false;

        RewritePatternSet patterns(context);

        patterns.add<
            // linalg operators
            MatMulPattern, GenericElementWisePattern, TransposePattern, BatchMatmulPattern,
            // custom dia operators
            DIAMatMulPattern, DIABatchMatMulPattern, DIAElementwisePattern, DIATransposePattern>(
            context, detectDIA);

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
                const BandedSubMatrix opBandInfo =
                    BandedStructureAnalysis::readPropertyFromDictAttr(metadata);
                if (opBandInfo.IsDia) changed = true;
            }

            if (!changed) return;

            auto newFuncType =
                FunctionType::get(context, funcOp.getFunctionType().getInputs(), newTypes);
            funcOp.setType(newFuncType);
        });
    }
};
}  // namespace mlir::bpa
