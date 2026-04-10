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

    LogicalResult denseTimesDenseToDiaBandedBatchMatmulToSCF(
        dia::BatchMatmulOp op, PatternRewriter& rewriter, const BandedSubMatrix& bandA,
        const BandedSubMatrix& bandB, const BandedSubMatrix& outputBand) const {
        Location loc = op.getLoc();
        Value A = op.getLhs();
        Value B = op.getRhs();

        auto resultType = cast<RankedTensorType>(op.getResult().getType());
        auto inputType = cast<RankedTensorType>(A.getType());

        const uint64_t batchSize = inputType.getDimSize(0);
        const uint64_t N = inputType.getDimSize(1);
        const uint64_t M = inputType.getDimSize(2);

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value dimBatch = arith::ConstantIndexOp::create(rewriter, loc, batchSize);
        Value dimN = arith::ConstantIndexOp::create(rewriter, loc, N);
        Value dimM = arith::ConstantIndexOp::create(rewriter, loc, M);

        Value lowerA = arith::ConstantIndexOp::create(rewriter, loc, bandA.Property.LowerBandwidth);
        Value upperA = arith::ConstantIndexOp::create(rewriter, loc, bandA.Property.UpperBandwidth);
        Value lowerB = arith::ConstantIndexOp::create(rewriter, loc, bandB.Property.LowerBandwidth);
        Value upperB = arith::ConstantIndexOp::create(rewriter, loc, bandB.Property.UpperBandwidth);

        // DIA output type: Batch x (lC + uC + 1) x N
        const int64_t lC = outputBand.Property.LowerBandwidth;
        const int64_t uC = outputBand.Property.UpperBandwidth;
        int64_t numDiags = lC + uC + 1;
        auto elementType = resultType.getElementType();
        auto diaType =
            RankedTensorType::get({ (int64_t)batchSize, numDiags, (int64_t)N }, elementType);

        Value emptyDia = tensor::EmptyOp::create(rewriter, loc, diaType, ValueRange{});
        Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                               rewriter.getZeroAttr(elementType));
        Value zeroedDia =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ emptyDia })
                .getResult(0);

        Value lCVal = arith::ConstantIndexOp::create(rewriter, loc, lC);

        auto bLoop = scf::ForOp::create(
            rewriter, loc, c0, dimBatch, c1, ValueRange{ zeroedDia },
            [&](OpBuilder& bb, Location loc, Value b, ValueRange bArgs) {
                Value cBatch = bArgs[0];

                auto iLoop = scf::ForOp::create(
                    bb, loc, c0, dimN, c1, ValueRange{ cBatch },
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

                                // k start = max(0, max(j - Ub, i - La))
                                Value iMinusLa = arith::SubIOp::create(mb, loc, i, lowerA);
                                Value jMinusUb = arith::SubIOp::create(mb, loc, j, upperB);
                                Value kStart0 = arith::MaxSIOp::create(mb, loc, jMinusUb, iMinusLa);
                                Value kStart = arith::MaxSIOp::create(mb, loc, c0, kStart0);

                                // k end = min(N, min(j + Lb, i + Ua) + 1)
                                Value iPlusUa = arith::AddIOp::create(mb, loc, i, upperA);
                                Value jPlusLb = arith::AddIOp::create(mb, loc, j, lowerB);
                                Value kEndPlusOne = arith::AddIOp::create(
                                    mb, loc, arith::MinSIOp::create(mb, loc, jPlusLb, iPlusUa), c1);
                                Value kEnd = arith::MinSIOp::create(mb, loc, kEndPlusOne, dimN);

                                auto kLoop = scf::ForOp::create(
                                    mb, loc, kStart, kEnd, c1, ValueRange{ cMid, initAcc },
                                    [&](OpBuilder& ib, Location loc, Value k,
                                        ValueRange innerArgs) {
                                        Value cInner = innerArgs[0];
                                        Value acc = innerArgs[1];

                                        // Extract 3D values {b, i, k} and {b, k, j}
                                        Value aVal = tensor::ExtractOp::create(
                                            ib, loc, A, ValueRange{ b, i, k });
                                        Value bVal = tensor::ExtractOp::create(
                                            ib, loc, B, ValueRange{ b, k, j });

                                        Value mul = arith::MulFOp::create(ib, loc, aVal, bVal);
                                        Value add = arith::AddFOp::create(ib, loc, acc, mul);

                                        scf::YieldOp::create(ib, loc, ValueRange{ cInner, add });
                                    });
                                // Write to DIA 3D: data[b, j - i + lC, i]
                                Value diagIdx = arith::AddIOp::create(
                                    mb, loc, arith::SubIOp::create(mb, loc, j, i), lCVal);
                                Value updated = tensor::InsertOp::create(
                                    mb, loc, kLoop.getResult(1), kLoop.getResult(0),
                                    ValueRange{ b, diagIdx, i });
                                scf::YieldOp::create(mb, loc, ValueRange{ updated });
                            });
                        scf::YieldOp::create(ob, loc, ValueRange{ jLoop.getResult(0) });
                    });
                scf::YieldOp::create(bb, loc, ValueRange{ iLoop.getResult(0) });
            });
        if (auto meta = op->getAttrOfType<DictionaryAttr>("metadata")) {
            bLoop->setAttr("metadata", meta);
        }
        rewriter.replaceOp(op, bLoop.getResult(0));
        return success();
    }
    LogicalResult diaTimesDenseToDiaBandedBatchMatmulToSCF(dia::BatchMatmulOp op,
                                                           PatternRewriter& rewriter,
                                                           const BandedSubMatrix& bandA,
                                                           const BandedSubMatrix& bandB) const {
        Location loc = op->getLoc();
        Value A = op.getLhs();
        Value B = op.getRhs();
        Value C = op.getOutput();
        auto resultType = cast<RankedTensorType>(C.getType());

        // Batch is dim 0, diags is dim 1, N is dim 2
        const int64_t batch = cast<RankedTensorType>(A.getType()).getDimSize(0);
        const int64_t N = cast<RankedTensorType>(B.getType()).getDimSize(2);

        const uint64_t upperA = bandA.Property.UpperBandwidth;
        const uint64_t lowerA = bandA.Property.LowerBandwidth;
        const uint64_t upperB = bandB.Property.UpperBandwidth;
        const uint64_t lowerB = bandB.Property.LowerBandwidth;

        int64_t olower = std::min(static_cast<uint64_t>(N - 1), lowerA + lowerB);
        int64_t oupper = std::min(static_cast<uint64_t>(N - 1), upperA + upperB);
        int64_t C_rows = olower + oupper + 1;

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value cN = arith::ConstantIndexOp::create(rewriter, loc, N);
        Value cBatch = arith::ConstantIndexOp::create(rewriter, loc, batch);
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

        // for b in range(batch)
        auto batchLoop = scf::ForOp::create(
            rewriter, loc, c0, cBatch, c1, ValueRange{ zeroedC },
            [&](OpBuilder& bb, Location loc, Value bIdx, ValueRange bArgs) {
                Value cOut = bArgs[0];

                // for d_C in range(C_rows)
                auto dLoop = scf::ForOp::create(
                    bb, loc, c0, cCRows, c1, ValueRange{ cOut },
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
                                Value colValid0 = arith::CmpIOp::create(
                                    rb, loc, arith::CmpIPredicate::sge, col, c0);
                                Value colValidN = arith::CmpIOp::create(
                                    rb, loc, arith::CmpIPredicate::slt, col, cN);
                                Value colValid =
                                    arith::AndIOp::create(rb, loc, colValid0, colValidN);

                                auto ifOp = scf::IfOp::create(rb, loc, TypeRange{ cRow.getType() },
                                                              colValid, true);

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
                                        [&](OpBuilder& kb, Location loc, Value k,
                                            ValueRange kArgs) {
                                            Value cK = kArgs[0];

                                            // d_A = (k - row) + lA
                                            Value kI64 = toI64(kb, k);
                                            Value rI64 = toI64(kb, row);
                                            Value dA = toIndex(
                                                kb, arith::AddIOp::create(
                                                        kb, loc,
                                                        arith::SubIOp::create(kb, loc, kI64, rI64),
                                                        cLAi64));

                                            // A[bIdx][d_A][row], B[bIdx][k][col]
                                            Value aVal = tensor::ExtractOp::create(
                                                kb, loc, A, ValueRange{ bIdx, dA, row });
                                            Value bVal = tensor::ExtractOp::create(
                                                kb, loc, B, ValueRange{ bIdx, k, col });

                                            // C[bIdx][d_C][row] += A[bIdx][d_A][row] *
                                            // B[bIdx][k][col]
                                            Value cVal = tensor::ExtractOp::create(
                                                kb, loc, cK, ValueRange{ bIdx, dC, row });
                                            Value mul = arith::MulFOp::create(kb, loc, aVal, bVal);
                                            Value acc = arith::AddFOp::create(kb, loc, cVal, mul);
                                            Value updated = tensor::InsertOp::create(
                                                kb, loc, acc, cK, ValueRange{ bIdx, dC, row });

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

                scf::YieldOp::create(bb, loc, dLoop.getResults());
            });

        batchLoop->setAttr("metadata", op->getAttrOfType<DictionaryAttr>("metadata"));
        rewriter.replaceOp(op, batchLoop.getResult(0));
        return success();
    }
    LogicalResult matchAndRewrite(dia::BatchMatmulOp op, PatternRewriter& rewriter) const override {
        auto dict = op->getAttrDictionary();
        if (!dict) dict = DictionaryAttr();

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
        const BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        if (!bandA.IsDia && !bandB.IsDia && opBandInfo.IsDia) {
            return denseTimesDenseToDiaBandedBatchMatmulToSCF(op, rewriter, bandA, bandB,
                                                              opBandInfo);
        } else if (bandA.IsDia && !bandB.IsDia && opBandInfo.IsDia) {
            return diaTimesDenseToDiaBandedBatchMatmulToSCF(op, rewriter, bandA, bandB);
            // } else if (bandA.IsDia && bandB.IsDia && !opBandInfo.IsDia) {
            //     return diaTimesDiaToDenseBandedBatchMatmulToSCF(op, rewriter, bandA, bandB,
            //     opBandInfo);
        } else
            return diaTimesDiaToDiaBandedBatchMatmulToSCF(op, rewriter, opBandInfo);
    }
};

void addDIABatchMatmulPatterns(RewritePatternSet& patterns) {
    patterns.add<DIABatchMatMulPattern>(patterns.getContext());
}

}  // namespace mlir::bpa
