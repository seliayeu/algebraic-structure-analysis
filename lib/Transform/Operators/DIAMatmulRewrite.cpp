
#include <cstdint>

#include "Analysis/BandedStructureAnalysis.h"
#include "Dialect/DIA/DIAOps.h"
#include "Utils/TransformUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

namespace mlir::bpa {

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

    LogicalResult diaTimesDenseToDiaBandedMatmulToSCF(dia::MatmulOp op,
                                                      PatternRewriter& rewriter) const {
        Location loc = op->getLoc();
        Value A = op.getLhs();
        Value B = op.getRhs();
        Value C = op.getOutput();
        auto resultType = cast<RankedTensorType>(C.getType());
        const int64_t N = cast<RankedTensorType>(B.getType()).getDimSize(1);

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

        dLoop->setAttr("metadata", op->getAttrOfType<DictionaryAttr>("metadata"));
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

    LogicalResult diaTimesDenseToDenseBandedMatmulToSCF(dia::MatmulOp op, PatternRewriter& rewriter,
                                                        const BandedSubMatrix& bandA,
                                                        const BandedSubMatrix& bandB) const {
        Location loc = op->getLoc();
        Value A = op.getLhs();
        Value B = op.getRhs();
        Value C = op.getOutput();

        auto resultType = cast<RankedTensorType>(C.getType());
        const int64_t N = resultType.getDimSize(1);

        Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        Value cN = arith::ConstantIndexOp::create(rewriter, loc, N);
        Value cLA = arith::ConstantIndexOp::create(rewriter, loc, bandA.Property.LowerBandwidth);
        Value cUA = arith::ConstantIndexOp::create(rewriter, loc, bandA.Property.UpperBandwidth);
        Value cLB = arith::ConstantIndexOp::create(rewriter, loc, bandB.Property.LowerBandwidth);
        Value cUB = arith::ConstantIndexOp::create(rewriter, loc, bandB.Property.UpperBandwidth);
        Value cLAi64 =
            arith::ConstantIntOp::create(rewriter, loc, bandA.Property.LowerBandwidth, 64);

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

        auto rowLoop = scf::ForOp::create(
            rewriter, loc, c0, cN, c1, ValueRange{ zeroedC },
            [&](OpBuilder& ob, Location loc, Value row, ValueRange rowArgs) {
                Value cRow = rowArgs[0];
                auto colLoop = scf::ForOp::create(
                    ob, loc, c0, cN, c1, ValueRange{ cRow },
                    [&](OpBuilder& cb, Location loc, Value col, ValueRange colArgs) {
                        Value cCol = colArgs[0];

                        // kstart = max(0, row - lA, col - uB)
                        Value rowMinusLA = arith::SubIOp::create(cb, loc, row, cLA);
                        Value colMinusUB = arith::SubIOp::create(cb, loc, col, cUB);
                        Value ks0 = arith::MaxSIOp::create(cb, loc, c0, rowMinusLA);
                        Value kstart = arith::MaxSIOp::create(cb, loc, ks0, colMinusUB);

                        // kend = min(N, row + uA + 1, col + lB + 1)
                        Value rowPlusUA1 = arith::AddIOp::create(
                            cb, loc, arith::AddIOp::create(cb, loc, row, cUA), c1);
                        Value colPlusLB1 = arith::AddIOp::create(
                            cb, loc, arith::AddIOp::create(cb, loc, col, cLB), c1);
                        Value ke0 = arith::MinSIOp::create(cb, loc, cN, rowPlusUA1);
                        Value kend = arith::MinSIOp::create(cb, loc, ke0, colPlusLB1);

                        auto kLoop = scf::ForOp::create(
                            cb, loc, kstart, kend, c1, ValueRange{ cCol },
                            [&](OpBuilder& kb, Location loc, Value k, ValueRange kArgs) {
                                Value cK = kArgs[0];

                                // d_A = (k - row) + lA
                                Value kI64 = toI64(kb, k);
                                Value rowI64 = toI64(kb, row);
                                Value dA = toIndex(
                                    kb, arith::AddIOp::create(
                                            kb, loc, arith::SubIOp::create(kb, loc, kI64, rowI64),
                                            cLAi64));

                                Value aVal =
                                    tensor::ExtractOp::create(kb, loc, A, ValueRange{ dA, row });
                                Value bVal =
                                    tensor::ExtractOp::create(kb, loc, B, ValueRange{ k, col });
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
                scf::YieldOp::create(ob, loc, colLoop.getResults());
            });

        if (auto metadata = op->getAttr("metadata")) rowLoop->setAttr("metadata", metadata);
        rewriter.replaceOp(op, rowLoop.getResult(0));
        return success();
    }
    // The linalg pattern rewritter will figure out about the input bands.
    // This is way this function doesn't follow the name pattern.
    LogicalResult denseTimesDenseToDenseMatmulToLinalg(dia::MatmulOp op, PatternRewriter& rewriter,
                                                       const BandedSubMatrix& bandA,
                                                       const BandedSubMatrix& bandB) const {
        auto A = op.getLhs();
        auto B = op.getRhs();
        auto C = op.getOutput();
        auto loc = op->getLoc();

        auto aType = cast<RankedTensorType>(A.getType());
        auto bType = cast<RankedTensorType>(B.getType());
        auto cType = cast<RankedTensorType>(C.getType());
        auto elementType = aType.getElementType();

        const int64_t N = aType.getDimSize(1);

        auto staticAType = RankedTensorType::get({ N, N }, elementType);
        auto staticBType = RankedTensorType::get({ N, N }, elementType);
        auto staticCType = RankedTensorType::get({ N, N }, elementType);

        Value castA = tensor::CastOp::create(rewriter, loc, staticAType, A).getResult();
        Value castB = tensor::CastOp::create(rewriter, loc, staticBType, B).getResult();
        Value castC = tensor::CastOp::create(rewriter, loc, staticCType, C).getResult();
        castA.getDefiningOp()->setAttr("metadata", bandA.toAttribute(rewriter));
        castB.getDefiningOp()->setAttr("metadata", bandB.toAttribute(rewriter));

        auto newOp = linalg::MatmulOp::create(rewriter, loc, TypeRange{ staticCType },
                                              ValueRange{ castA, castB }, ValueRange{ castC });

        if (auto metadata = op->getAttr("metadata")) newOp->setAttr("metadata", metadata);

        rewriter.replaceOp(op, newOp);
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
            else if (!bandA.IsDia && !bandB.IsDia && !resultBand.IsDia)
                return denseTimesDenseToDenseMatmulToLinalg(op, rewriter, bandA, bandB);
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
                return diaTimesDenseToDiaBandedMatmulToSCF(op, rewriter);
            } else if (!bandA.IsDia && !bandB.IsDia && resultBand.IsDia) {
                return denseTimesDenseToDiaBandedMatmulToSCF(op, rewriter, resultBand);
            } else if (bandA.IsDia && !bandB.IsDia && !resultBand.IsDia) {
                // TODO: create a lowering in case a is diagonal
                return diaTimesDenseToDenseBandedMatmulToSCF(op, rewriter, bandA, bandB);
            } else if (!bandA.IsDia && !bandB.IsDia && !resultBand.IsDia) {
                // this op is already implemented in the linalg lowering.
                return denseTimesDenseToDenseMatmulToLinalg(op, rewriter, bandA, bandB);
            }
            return diaTimesDiaToDiaBandedMatmulToSCF(op, rewriter, resultBand);
        }
    }

   private:
    bool detectDIA;
};

void addDIAMatmulPatterns(RewritePatternSet& patterns, bool detectDia) {
    patterns.add<DIAMatMulPattern>(patterns.getContext(), detectDia);
}

}  // namespace mlir::bpa
