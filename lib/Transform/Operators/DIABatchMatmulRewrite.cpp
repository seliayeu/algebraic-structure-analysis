#include <cstdint>

#include "Analysis/BandedStructureAnalysis.h"
#include "Dialect/DIA/DIAOps.h"
#include "Utils/TransformUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace mlir::bpa {

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

void addDIABatchMatmulPatterns(RewritePatternSet& patterns) {
    patterns.add<DIABatchMatMulPattern>(patterns.getContext());
}

}  // namespace mlir::bpa
