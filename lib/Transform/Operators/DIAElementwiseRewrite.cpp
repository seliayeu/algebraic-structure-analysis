#include <cstdint>
#include <iostream>

#include "Analysis/BandedStructureAnalysis.h"
#include "Dialect/DIA/DIAOps.h"
#include "Utils/TransformUtils.h"
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
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

namespace mlir::bpa {

struct DIAElementwisePattern : public OpRewritePattern<dia::ElementwiseOp> {
private:
    Value diaTimesDiaToDenseKernel(
        OpBuilder& rewriter, Location loc, dia::ElementwiseOp op,
        Value A, Value B, Value initC, Type elementType,
        Value totalRows, Value totalCols,
        int64_t lA, int64_t lB, int64_t uA, int64_t uB,
        ArrayRef<Value> batchIndices) const {

        auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        auto cf0 = arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(elementType, 0.0));

        auto lDiff = arith::ConstantIndexOp::create(rewriter, loc, std::max(lA, lB) - std::min(lA, lB));
        auto lMax = arith::ConstantIndexOp::create(rewriter, loc, std::max(lA, lB));
        auto lMin = arith::ConstantIndexOp::create(rewriter, loc, std::min(lA, lB));
        auto uDiff = arith::ConstantIndexOp::create(rewriter, loc, std::max(uA, uB) - std::min(uA, uB));
        
        auto cLA = arith::ConstantIndexOp::create(rewriter, loc, lA);
        auto cLB = arith::ConstantIndexOp::create(rewriter, loc, lB);

        Value currC = initC;

        if (op.getKind() != dia::ElementwiseKind::mul && (lA > lB || lB > lA)) {
            scf::ForOp iLoop = scf::ForOp::create(
                rewriter, loc, c0, lDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::SubIOp::create(ob, loc, lMax, i);
                    Value mMinusBand = arith::SubIOp::create(ob, loc, totalRows, currBand);
                    Value numEls = arith::MinUIOp::create(ob, loc, mMinusBand, totalCols);
                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cRow = arith::AddIOp::create(ib, loc, r, currBand);
                            Value operand1, operand2;

                            SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                            idxA.push_back(i);
                            idxA.push_back(cRow);

                            SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                            idxB.push_back(i);
                            idxB.push_back(cRow);

                            if (lA > lB) {
                                operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA);
                                operand2 = cf0;
                            } else {
                                operand1 = cf0;
                                operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB);
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

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(cRow);
                            idxC.push_back(r);

                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoop.getResult(0);
        }

        if (op.getKind() != dia::ElementwiseKind::mul && (uA > uB || uB > uA)) {
            auto startUpperBand = arith::ConstantIndexOp::create(rewriter, loc, std::min(uA, uB) + 1);
            scf::ForOp iLoop = scf::ForOp::create(
                rewriter, loc, c0, uDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::AddIOp::create(ob, loc, i, startUpperBand);
                    Value aInd, bInd;
                    if (uA > uB) {
                        aInd = arith::AddIOp::create(ob, loc, currBand, cLA);
                    } else {
                        bInd = arith::AddIOp::create(ob, loc, currBand, cLB);
                    }
                    Value nMinusBand = arith::SubIOp::create(ob, loc, totalCols, currBand);
                    Value numEls = arith::MinUIOp::create(ob, loc, totalRows, nMinusBand);
                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cCol = arith::AddIOp::create(ib, loc, r, currBand);
                            Value operand1, operand2;

                            if (uA > uB) {
                                SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                                idxA.push_back(aInd);
                                idxA.push_back(r);
                                operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA);
                                operand2 = cf0;
                            } else {
                                SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                                idxB.push_back(bInd);
                                idxB.push_back(r);
                                operand1 = cf0;
                                operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB);
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

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(r);
                            idxC.push_back(cCol);

                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoop.getResult(0);
        }

        auto totalLowerDiags = arith::ConstantIndexOp::create(rewriter, loc, std::min(lA, lB) + 1);
        scf::ForOp iLoopLower = scf::ForOp::create(
            rewriter, loc, c0, totalLowerDiags, c1, ValueRange{ currC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                auto cOuter = iArgs[0];
                Value aInd = (lA > lB) ? arith::AddIOp::create(ob, loc, i, lDiff).getResult() : i;
                Value bInd = (lA > lB) ? i : arith::AddIOp::create(ob, loc, i, lDiff).getResult();

                Value currBand = arith::SubIOp::create(ob, loc, lMin, i);
                Value mMinusBand = arith::SubIOp::create(ob, loc, totalRows, currBand);
                Value numEls = arith::MinUIOp::create(ob, loc, mMinusBand, totalCols);
                auto rLoop = scf::ForOp::create(
                    ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                    [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                        auto cInner = rArgs[0];
                        Value cRow = arith::AddIOp::create(ib, loc, r, currBand);

                        SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                        idxA.push_back(aInd);
                        idxA.push_back(cRow);
                        auto operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA);

                        SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                        idxB.push_back(bInd);
                        idxB.push_back(cRow);
                        auto operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB);

                        Value newOp;
                        switch (op.getKind()) {
                            case dia::ElementwiseKind::add:
                                newOp = arith::AddFOp::create(ib, loc, operand1, operand2);
                                break;
                            case dia::ElementwiseKind::sub:
                                newOp = arith::SubFOp::create(ib, loc, operand1, operand2);
                                break;
                            case dia::ElementwiseKind::mul:
                                newOp = arith::MulFOp::create(ib, loc, operand1, operand2);
                                break;
                            default:
                                assert(false);
                        }

                        SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                        idxC.push_back(cRow);
                        idxC.push_back(r);

                        auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    });
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            });
        currC = iLoopLower.getResult(0);

        if (std::min(uA, uB) != 0) {
            auto totalUpperDiags = arith::ConstantIndexOp::create(rewriter, loc, std::min(uA, uB) + 1);
            auto iLoopUpper = scf::ForOp::create(
                rewriter, loc, c1, totalUpperDiags, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value aInd = arith::AddIOp::create(ob, loc, i, cLA);
                    Value bInd = arith::AddIOp::create(ob, loc, i, cLB);
                    Value nMinusBand = arith::SubIOp::create(ob, loc, totalCols, i);
                    Value numEls = arith::MinUIOp::create(ob, loc, totalRows, nMinusBand);
                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cCol = arith::AddIOp::create(ib, loc, r, i);

                            SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                            idxA.push_back(aInd);
                            idxA.push_back(r);
                            auto operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA);

                            SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                            idxB.push_back(bInd);
                            idxB.push_back(r);
                            auto operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB);

                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2);
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2);
                                    break;
                                case dia::ElementwiseKind::mul:
                                    newOp = arith::MulFOp::create(ib, loc, operand1, operand2);
                                    break;
                                default:
                                    assert(false);
                            }

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(r);
                            idxC.push_back(cCol);

                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoopUpper.getResult(0);
        }

        return currC;
    }

    Value diaTimesDiaToDiaKernel(
        OpBuilder& rewriter, Location loc, dia::ElementwiseOp op,
        Value A, Value B, Value initC, Type elementType,
        Value totalRows, int64_t lA, int64_t lB, int64_t uA, int64_t uB, int64_t lC,
        ArrayRef<Value> batchIndices) const {

        auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        auto cf0 = arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(elementType, 0.0));

        auto cLA = arith::ConstantIndexOp::create(rewriter, loc, lA);
        auto cLB = arith::ConstantIndexOp::create(rewriter, loc, lB);
        auto cLC = arith::ConstantIndexOp::create(rewriter, loc, lC);

        Value currC = initC;

        if (op.getKind() != dia::ElementwiseKind::mul && (lA > lB || lB > lA)) {
            auto totalDiags = arith::ConstantIndexOp::create(rewriter, loc, std::max(lA, lB) - std::min(lA, lB));
            auto lMax = arith::ConstantIndexOp::create(rewriter, loc, std::max(lA, lB));

            scf::ForOp iLoop = scf::ForOp::create(
                rewriter, loc, c0, totalDiags, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::SubIOp::create(ob, loc, i, lMax).getResult();
                    Value cInd = arith::AddIOp::create(ob, loc, cLC, currBand).getResult();

                    Value aInd, bInd;
                    if (lA > lB)
                        aInd = arith::AddIOp::create(ob, loc, cLA, currBand).getResult();
                    else
                        bInd = arith::AddIOp::create(ob, loc, cLB, currBand).getResult();

                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, totalRows, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            Value operand1 = cf0;
                            if (lA > lB) {
                                SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                                idxA.push_back(aInd);
                                idxA.push_back(r);
                                operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();
                            }

                            Value operand2 = cf0;
                            if (lB > lA) {
                                SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                                idxB.push_back(bInd);
                                idxB.push_back(r);
                                operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();
                            }

                            Value newOp;
                            if (op.getKind() == dia::ElementwiseKind::add)
                                newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                            else
                                newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(cInd);
                            idxC.push_back(r);

                            auto updated = tensor::InsertOp::create(ib, loc, newOp, rArgs[0], idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoop.getResult(0);
        }

        if (op.getKind() != dia::ElementwiseKind::mul && (uA > uB || uB > uA)) {
            auto numDiagonals = arith::ConstantIndexOp::create(rewriter, loc, std::max(uA, uB) - std::min(uA, uB));
            auto startUpperBand = arith::ConstantIndexOp::create(rewriter, loc, std::min(uA, uB) + 1);

            scf::ForOp iLoop = scf::ForOp::create(
                rewriter, loc, c0, numDiagonals, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::AddIOp::create(ob, loc, i, startUpperBand).getResult();
                    Value cInd = arith::AddIOp::create(ob, loc, cLC, currBand).getResult();

                    Value aInd, bInd;
                    if (uA > uB)
                        aInd = arith::AddIOp::create(ob, loc, cLA, currBand).getResult();
                    else
                        bInd = arith::AddIOp::create(ob, loc, cLB, currBand).getResult();

                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, totalRows, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            Value operand1 = cf0;
                            if (uA > uB) {
                                SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                                idxA.push_back(aInd);
                                idxA.push_back(r);
                                operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();
                            }

                            Value operand2 = cf0;
                            if (uB > uA) {
                                SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                                idxB.push_back(bInd);
                                idxB.push_back(r);
                                operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();
                            }

                            Value newOp;
                            if (op.getKind() == dia::ElementwiseKind::add)
                                newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                            else
                                newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(cInd);
                            idxC.push_back(r);

                            auto updated = tensor::InsertOp::create(ib, loc, newOp, rArgs[0], idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoop.getResult(0);
        }

        auto numDiagonals = arith::ConstantIndexOp::create(rewriter, loc, std::min(lA, lB) + 1 + std::min(uA, uB));
        auto minL = arith::ConstantIndexOp::create(rewriter, loc, std::min(lA, lB));

        scf::ForOp iLoop = scf::ForOp::create(
            rewriter, loc, c0, numDiagonals, c1, ValueRange{ currC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                auto cOuter = iArgs[0];
                Value currBand = arith::SubIOp::create(ob, loc, i, minL).getResult();
                Value aInd = arith::AddIOp::create(ob, loc, cLA, currBand).getResult();
                Value bInd = arith::AddIOp::create(ob, loc, cLB, currBand).getResult();
                Value cInd = arith::AddIOp::create(ob, loc, cLC, currBand).getResult();

                auto rLoop = scf::ForOp::create(
                    ob, loc, c0, totalRows, c1, ValueRange{ cOuter },
                    [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                        SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                        idxA.push_back(aInd);
                        idxA.push_back(r);
                        Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();

                        SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                        idxB.push_back(bInd);
                        idxB.push_back(r);
                        Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();

                        Value newOp;
                        switch (op.getKind()) {
                            case dia::ElementwiseKind::add:
                                newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            case dia::ElementwiseKind::sub:
                                newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            case dia::ElementwiseKind::mul:
                                newOp = arith::MulFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            default:
                                assert(false);
                        }

                        SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                        idxC.push_back(cInd);
                        idxC.push_back(r);

                        auto updated = tensor::InsertOp::create(ib, loc, newOp, rArgs[0], idxC);
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    });
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            });

        return iLoop.getResult(0);
    }

    Value diaTimesDenseToDiaKernel(
        OpBuilder& rewriter, Location loc, dia::ElementwiseOp op,
        Value A, Value B, Value initC, Type elementType,
        Value totalRows, Value totalCols,
        int64_t lA, int64_t uA, int64_t lC, int64_t uC,
        ArrayRef<Value> batchIndices) const {

        auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        auto cf0 = arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(elementType, 0.0));

        auto cLA = arith::ConstantIndexOp::create(rewriter, loc, lA);
        auto cLC = arith::ConstantIndexOp::create(rewriter, loc, lC);

        Value currC = initC;

        if (lC > lA) {
            auto lDiff = arith::ConstantIndexOp::create(rewriter, loc, lC - lA);
            scf::ForOp iLoop = scf::ForOp::create(
                rewriter, loc, c0, lDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::SubIOp::create(ob, loc, cLC, i).getResult();
                    Value cInd = arith::SubIOp::create(ob, loc, cLC, currBand).getResult();

                    Value mMinusBand = arith::SubIOp::create(ob, loc, totalRows, currBand).getResult();
                    Value numEls = arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult();

                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cRow = arith::AddIOp::create(ib, loc, r, currBand).getResult();
                            
                            Value operand1 = cf0;
                            
                            SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                            idxB.push_back(cRow);
                            idxB.push_back(r);
                            Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();

                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::mul:
                                    newOp = arith::MulFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                default:
                                    assert(false);
                            }
                            
                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(cInd);
                            idxC.push_back(cRow);

                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoop.getResult(0);
        }

        auto lMinVal = std::min(lA, lC);
        if (lMinVal > 0) {
            auto totalLowerDiags = arith::ConstantIndexOp::create(rewriter, loc, lMinVal);
            auto lMin = arith::ConstantIndexOp::create(rewriter, loc, lMinVal);
            scf::ForOp iLoopLower = scf::ForOp::create(
                rewriter, loc, c0, totalLowerDiags, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::SubIOp::create(ob, loc, lMin, i).getResult();
                    Value aInd = arith::SubIOp::create(ob, loc, cLA, currBand).getResult();
                    Value cInd = arith::SubIOp::create(ob, loc, cLC, currBand).getResult();

                    Value mMinusBand = arith::SubIOp::create(ob, loc, totalRows, currBand).getResult();
                    Value numEls = arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult();

                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cRow = arith::AddIOp::create(ib, loc, r, currBand).getResult();

                            SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                            idxA.push_back(aInd);
                            idxA.push_back(cRow);
                            Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();
                            
                            SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                            idxB.push_back(cRow);
                            idxB.push_back(r);
                            Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();

                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::mul:
                                    newOp = arith::MulFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                default:
                                    assert(false);
                            }

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(cInd);
                            idxC.push_back(cRow);

                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoopLower.getResult(0);
        }

        auto uMinVal = std::min(uA, uC);
        auto totalUpperDiags = arith::ConstantIndexOp::create(rewriter, loc, uMinVal + 1);
        scf::ForOp iLoopUpper = scf::ForOp::create(
            rewriter, loc, c0, totalUpperDiags, c1, ValueRange{ currC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                auto cOuter = iArgs[0];
                Value currBand = i;
                Value aInd = arith::AddIOp::create(ob, loc, cLA, currBand).getResult();
                Value cInd = arith::AddIOp::create(ob, loc, cLC, currBand).getResult();

                Value nMinusBand = arith::SubIOp::create(ob, loc, totalCols, currBand).getResult();
                Value numEls = arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult();

                auto rLoop = scf::ForOp::create(
                    ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                    [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                        auto cInner = rArgs[0];
                        Value cCol = arith::AddIOp::create(ib, loc, r, currBand).getResult();

                        SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                        idxA.push_back(aInd);
                        idxA.push_back(r);
                        Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();
                        
                        SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                        idxB.push_back(r);
                        idxB.push_back(cCol);
                        Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();

                        Value newOp;
                        switch (op.getKind()) {
                            case dia::ElementwiseKind::add:
                                newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            case dia::ElementwiseKind::sub:
                                newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            case dia::ElementwiseKind::mul:
                                newOp = arith::MulFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            default:
                                assert(false);
                        }

                        SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                        idxC.push_back(cInd);
                        idxC.push_back(r);

                        auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    });
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            });
        currC = iLoopUpper.getResult(0);

        if (uC > uA) {
            auto uDiff = arith::ConstantIndexOp::create(rewriter, loc, uC - uA);
            auto startUpperBand = arith::ConstantIndexOp::create(rewriter, loc, uA + 1);
            scf::ForOp iLoop = scf::ForOp::create(
                rewriter, loc, c0, uDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::AddIOp::create(ob, loc, i, startUpperBand).getResult();
                    Value cInd = arith::AddIOp::create(ob, loc, cLC, currBand).getResult();

                    Value nMinusBand = arith::SubIOp::create(ob, loc, totalCols, currBand).getResult();
                    Value numEls = arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult();

                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cCol = arith::AddIOp::create(ib, loc, r, currBand).getResult();
                            
                            Value operand1 = cf0; 
                            
                            SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                            idxB.push_back(r);
                            idxB.push_back(cCol);
                            Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();

                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::mul:
                                    newOp = arith::MulFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                default:
                                    assert(false);
                            }

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(cInd);
                            idxC.push_back(r);

                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoop.getResult(0);
        }

        return currC;
    }

    Value denseTimesDenseToDiaKernel(
        OpBuilder& rewriter, Location loc, dia::ElementwiseOp op,
        Value A, Value B, Value initC, Type elementType,
        Value totalRows, Value totalCols, int64_t lC, int64_t uC,
        ArrayRef<Value> batchIndices) const {

        auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        auto cLC = arith::ConstantIndexOp::create(rewriter, loc, lC);

        Value currC = initC;

        auto totalLowerDiags = arith::ConstantIndexOp::create(rewriter, loc, lC + 1);
        auto iLoopLower = scf::ForOp::create(
            rewriter, loc, c0, totalLowerDiags, c1, ValueRange{ currC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                auto cOuter = iArgs[0];
                Value currBand = arith::SubIOp::create(ob, loc, cLC, i).getResult();
                Value cInd = i; 

                Value mMinusBand = arith::SubIOp::create(ob, loc, totalRows, currBand).getResult();
                Value numEls = arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult();

                auto rLoop = scf::ForOp::create(
                    ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                    [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                        auto cInner = rArgs[0];
                        Value cRow = arith::AddIOp::create(ib, loc, r, currBand).getResult();

                        SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                        idxA.push_back(cRow);
                        idxA.push_back(r);
                        Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();

                        SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                        idxB.push_back(cRow);
                        idxB.push_back(r);
                        Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();

                        Value newOp;
                        switch (op.getKind()) {
                            case dia::ElementwiseKind::add:
                                newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            case dia::ElementwiseKind::sub:
                                newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            case dia::ElementwiseKind::mul:
                                newOp = arith::MulFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            default:
                                assert(false && "Unsupported elementwise kind");
                        }

                        SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                        idxC.push_back(cInd);
                        idxC.push_back(cRow);

                        auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    });
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            });
        currC = iLoopLower.getResult(0);

        if (uC > 0) {
            auto totalUpperDiags = arith::ConstantIndexOp::create(rewriter, loc, uC);
            auto iLoopUpper = scf::ForOp::create(
                rewriter, loc, c0, totalUpperDiags, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::AddIOp::create(ob, loc, i, c1).getResult();
                    Value cInd = arith::AddIOp::create(ob, loc, cLC, currBand).getResult();

                    Value nMinusBand = arith::SubIOp::create(ob, loc, totalCols, currBand).getResult();
                    Value numEls = arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult();

                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cCol = arith::AddIOp::create(ib, loc, r, currBand).getResult();

                            SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                            idxA.push_back(r);
                            idxA.push_back(cCol);
                            Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();

                            SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                            idxB.push_back(r);
                            idxB.push_back(cCol);
                            Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();

                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::mul:
                                    newOp = arith::MulFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                default:
                                    assert(false && "Unsupported elementwise kind");
                            }

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(cInd);
                            idxC.push_back(r);

                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoopUpper.getResult(0);
        }

        return currC;
    }

    Value denseTimesDiaToDiaKernel(
        OpBuilder& rewriter, Location loc, dia::ElementwiseOp op,
        Value A, Value B, Value initC, Type elementType,
        Value totalRows, Value totalCols,
        int64_t lB, int64_t uB, int64_t lC, int64_t uC,
        ArrayRef<Value> batchIndices) const {

        auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        auto cf0 = arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(elementType, 0.0));

        auto cLB = arith::ConstantIndexOp::create(rewriter, loc, lB);
        auto cLC = arith::ConstantIndexOp::create(rewriter, loc, lC);

        Value currC = initC;

        if (lC > lB) {
            auto lDiff = arith::ConstantIndexOp::create(rewriter, loc, lC - lB);
            scf::ForOp iLoop = scf::ForOp::create(
                rewriter, loc, c0, lDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::SubIOp::create(ob, loc, cLC, i).getResult();
                    Value cInd = arith::SubIOp::create(ob, loc, cLC, currBand).getResult();

                    Value mMinusBand = arith::SubIOp::create(ob, loc, totalRows, currBand).getResult();
                    Value numEls = arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult();

                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cRow = arith::AddIOp::create(ib, loc, r, currBand).getResult();
                            
                            SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                            idxA.push_back(cRow);
                            idxA.push_back(r);
                            Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();
                            
                            Value operand2 = cf0;

                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::mul:
                                    newOp = arith::MulFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                default:
                                    assert(false);
                            }

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(cInd);
                            idxC.push_back(cRow);

                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoop.getResult(0);
        }

        auto lMinVal = std::min(lB, lC);
        if (lMinVal > 0) {
            auto totalLowerDiags = arith::ConstantIndexOp::create(rewriter, loc, lMinVal);
            auto lMin = arith::ConstantIndexOp::create(rewriter, loc, lMinVal);
            scf::ForOp iLoopLower = scf::ForOp::create(
                rewriter, loc, c0, totalLowerDiags, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::SubIOp::create(ob, loc, lMin, i).getResult();
                    Value bInd = arith::SubIOp::create(ob, loc, cLB, currBand).getResult();
                    Value cInd = arith::SubIOp::create(ob, loc, cLC, currBand).getResult();

                    Value mMinusBand = arith::SubIOp::create(ob, loc, totalRows, currBand).getResult();
                    Value numEls = arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult();

                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cRow = arith::AddIOp::create(ib, loc, r, currBand).getResult();

                            SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                            idxA.push_back(cRow);
                            idxA.push_back(r);
                            Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();
                            
                            SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                            idxB.push_back(bInd);
                            idxB.push_back(cRow);
                            Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();

                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::mul:
                                    newOp = arith::MulFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                default:
                                    assert(false);
                            }

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(cInd);
                            idxC.push_back(cRow);

                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoopLower.getResult(0);
        }

        auto uMinVal = std::min(uB, uC);
        auto totalUpperDiags = arith::ConstantIndexOp::create(rewriter, loc, uMinVal + 1);
        scf::ForOp iLoopUpper = scf::ForOp::create(
            rewriter, loc, c0, totalUpperDiags, c1, ValueRange{ currC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                auto cOuter = iArgs[0];
                Value currBand = i;
                Value bInd = arith::AddIOp::create(ob, loc, cLB, currBand).getResult();
                Value cInd = arith::AddIOp::create(ob, loc, cLC, currBand).getResult();

                Value nMinusBand = arith::SubIOp::create(ob, loc, totalCols, currBand).getResult();
                Value numEls = arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult();

                auto rLoop = scf::ForOp::create(
                    ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                    [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                        auto cInner = rArgs[0];
                        Value cCol = arith::AddIOp::create(ib, loc, r, currBand).getResult();

                        SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                        idxA.push_back(r);
                        idxA.push_back(cCol);
                        Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();
                        
                        SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                        idxB.push_back(bInd);
                        idxB.push_back(r);
                        Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();

                        Value newOp;
                        switch (op.getKind()) {
                            case dia::ElementwiseKind::add:
                                newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            case dia::ElementwiseKind::sub:
                                newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            case dia::ElementwiseKind::mul:
                                newOp = arith::MulFOp::create(ib, loc, operand1, operand2).getResult();
                                break;
                            default:
                                assert(false);
                        }

                        SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                        idxC.push_back(cInd);
                        idxC.push_back(r);

                        auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    });
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            });
        currC = iLoopUpper.getResult(0);

        if (uC > uB) {
            auto uDiff = arith::ConstantIndexOp::create(rewriter, loc, uC - uB);
            auto startUpperBand = arith::ConstantIndexOp::create(rewriter, loc, uB + 1);
            scf::ForOp iLoop = scf::ForOp::create(
                rewriter, loc, c0, uDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::AddIOp::create(ob, loc, i, startUpperBand).getResult();
                    Value cInd = arith::AddIOp::create(ob, loc, cLC, currBand).getResult();

                    Value nMinusBand = arith::SubIOp::create(ob, loc, totalCols, currBand).getResult();
                    Value numEls = arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult();

                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cCol = arith::AddIOp::create(ib, loc, r, currBand).getResult();
                            
                            SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                            idxA.push_back(r);
                            idxA.push_back(cCol);
                            Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();
                            
                            Value operand2 = cf0; 

                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                case dia::ElementwiseKind::mul:
                                    newOp = arith::MulFOp::create(ib, loc, operand1, operand2).getResult();
                                    break;
                                default:
                                    assert(false);
                            }

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(cInd);
                            idxC.push_back(r);

                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoop.getResult(0);
        }

        return currC;
    }

    Value diaToDiaKernel(OpBuilder& rewriter, Location loc, Value input, Value initC,
                         linalg::ElementwiseKind linalgKind) const {
        
        MLIRContext* context = rewriter.getContext();
        auto inputType = cast<RankedTensorType>(input.getType());
        int64_t rank = inputType.getRank();
        
        AffineMap identityMap = rewriter.getMultiDimIdentityMap(rank);
        auto indexingMapsAttr = rewriter.getAffineMapArrayAttr({ identityMap, identityMap });

        SmallVector<Attribute> iteratorTypes(
            rank, linalg::IteratorTypeAttr::get(context, utils::IteratorType::parallel));
        auto iteratorTypesAttr = rewriter.getArrayAttr(iteratorTypes);

        SmallVector<NamedAttribute> attrs{
            rewriter.getNamedAttr("kind", linalg::ElementwiseKindAttr::get(context, linalgKind)),
            rewriter.getNamedAttr("indexing_maps", indexingMapsAttr),
            rewriter.getNamedAttr("iterator_types", iteratorTypesAttr)
        };

        auto elementwiseOp = linalg::ElementwiseOp::create(
            rewriter, loc, ValueRange{ input }, ValueRange{ initC }, attrs);
            
        return elementwiseOp->getResult(0);
    }

    Value diaTimesDenseToDenseKernel(
        OpBuilder& rewriter, Location loc, dia::ElementwiseOp op,
        Value A, Value B, Value initC, Type elementType,
        Value totalRows, Value totalCols,
        int64_t lA, int64_t uA, int64_t lC, int64_t uC,
        ArrayRef<Value> batchIndices) const {

        auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        auto cf0 = arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(elementType, 0.0));

        auto cLA = arith::ConstantIndexOp::create(rewriter, loc, lA);
        auto cLC = arith::ConstantIndexOp::create(rewriter, loc, lC);

        Value currC = initC;

        if (lC > lA) {
            auto lDiff = arith::ConstantIndexOp::create(rewriter, loc, lC - lA);
            scf::ForOp iLoop = scf::ForOp::create(
                rewriter, loc, c0, lDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::SubIOp::create(ob, loc, cLC, i).getResult();
                    Value mMinusBand = arith::SubIOp::create(ob, loc, totalRows, currBand).getResult();
                    Value numEls = arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult();

                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cRow = arith::AddIOp::create(ib, loc, r, currBand).getResult();
                            
                            Value operand1 = cf0;

                            SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                            idxB.push_back(cRow);
                            idxB.push_back(r);
                            Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();

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

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(cRow);
                            idxC.push_back(r);

                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoop.getResult(0);
        }

        auto lMinVal = std::min(lA, lC);
        auto totalLowerDiags = arith::ConstantIndexOp::create(rewriter, loc, lMinVal + 1); // do main diag
        auto lMin = arith::ConstantIndexOp::create(rewriter, loc, lMinVal);

        scf::ForOp iLoopLower = scf::ForOp::create(
            rewriter, loc, c0, totalLowerDiags, c1, ValueRange{ currC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                auto cOuter = iArgs[0];
                Value currBand = arith::SubIOp::create(ob, loc, lMin, i).getResult();
                Value aInd = arith::SubIOp::create(ob, loc, cLA, currBand).getResult();
                Value mMinusBand = arith::SubIOp::create(ob, loc, totalRows, currBand).getResult();
                Value numEls = arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult();

                auto rLoop = scf::ForOp::create(
                    ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                    [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                        auto cInner = rArgs[0];
                        Value cRow = arith::AddIOp::create(ib, loc, r, currBand).getResult();

                        SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                        idxA.push_back(aInd);
                        idxA.push_back(cRow);
                        Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();

                        SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                        idxB.push_back(cRow);
                        idxB.push_back(r);
                        Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();

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

                        SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                        idxC.push_back(cRow);
                        idxC.push_back(r);

                        auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    });
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            });
        currC = iLoopLower.getResult(0);

        if (uC > uA) {
            auto uDiff = arith::ConstantIndexOp::create(rewriter, loc, uC - uA);
            auto startUpperBand = arith::ConstantIndexOp::create(rewriter, loc, uA + 1);
            scf::ForOp iLoop = scf::ForOp::create(
                rewriter, loc, c0, uDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::AddIOp::create(ob, loc, i, startUpperBand).getResult();
                    Value nMinusBand = arith::SubIOp::create(ob, loc, totalCols, currBand).getResult();
                    Value numEls = arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult();

                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cCol = arith::AddIOp::create(ib, loc, r, currBand).getResult();
                            
                            Value operand1 = cf0;

                            SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                            idxB.push_back(r);
                            idxB.push_back(cCol);
                            Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();

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

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(r);
                            idxC.push_back(cCol);

                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoop.getResult(0);
        }

        auto uMinVal = std::min(uA, uC);
        if (uMinVal != 0) {
            auto totalUpperDiags = arith::ConstantIndexOp::create(rewriter, loc, uMinVal);
            scf::ForOp iLoopUpper = scf::ForOp::create(
                rewriter, loc, c0, totalUpperDiags, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::AddIOp::create(ob, loc, i, c1).getResult();
                    Value aInd = arith::AddIOp::create(ob, loc, cLA, currBand).getResult();
                    Value nMinusBand = arith::SubIOp::create(ob, loc, totalCols, currBand).getResult();
                    Value numEls = arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult();

                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cCol = arith::AddIOp::create(ib, loc, r, currBand).getResult();

                            SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                            idxA.push_back(aInd);
                            idxA.push_back(r);
                            Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();

                            SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                            idxB.push_back(r);
                            idxB.push_back(cCol);
                            Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();

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

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(r);
                            idxC.push_back(cCol);

                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoopUpper.getResult(0);
        }

        return currC;
    }

    Value denseTimesDiaToDenseKernel(
        OpBuilder& rewriter, Location loc, dia::ElementwiseOp op,
        Value A, Value B, Value initC, Type elementType,
        Value totalRows, Value totalCols,
        int64_t lB, int64_t uB, int64_t lC, int64_t uC,
        ArrayRef<Value> batchIndices) const {

        auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
        auto cf0 = arith::ConstantOp::create(rewriter, loc, rewriter.getFloatAttr(elementType, 0.0));

        auto cLB = arith::ConstantIndexOp::create(rewriter, loc, lB);
        auto cLC = arith::ConstantIndexOp::create(rewriter, loc, lC);

        Value currC = initC;

        if (lC > lB) {
            auto lDiff = arith::ConstantIndexOp::create(rewriter, loc, lC - lB);
            scf::ForOp iLoop = scf::ForOp::create(
                rewriter, loc, c0, lDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::SubIOp::create(ob, loc, cLC, i).getResult();
                    Value mMinusBand = arith::SubIOp::create(ob, loc, totalRows, currBand).getResult();
                    Value numEls = arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult();
                    
                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cRow = arith::AddIOp::create(ib, loc, r, currBand).getResult();
                            
                            SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                            idxA.push_back(cRow);
                            idxA.push_back(r);
                            Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();
                            
                            Value operand2 = cf0; // B is outside its lower bandwidth
                            
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

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(cRow);
                            idxC.push_back(r);
                            
                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoop.getResult(0);
        }

        auto lMinVal = std::min(lB, lC);
        auto totalLowerDiags = arith::ConstantIndexOp::create(rewriter, loc, lMinVal + 1);
        auto lMin = arith::ConstantIndexOp::create(rewriter, loc, lMinVal);
        scf::ForOp iLoopLower = scf::ForOp::create(
            rewriter, loc, c0, totalLowerDiags, c1, ValueRange{ currC },
            [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                auto cOuter = iArgs[0];
                Value currBand = arith::SubIOp::create(ob, loc, lMin, i).getResult();
                Value bInd = arith::SubIOp::create(ob, loc, cLB, currBand).getResult();
                Value mMinusBand = arith::SubIOp::create(ob, loc, totalRows, currBand).getResult();
                Value numEls = arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult();
                
                auto rLoop = scf::ForOp::create(
                    ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                    [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                        auto cInner = rArgs[0];
                        Value cRow = arith::AddIOp::create(ib, loc, r, currBand).getResult();
                        
                        SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                        idxA.push_back(cRow);
                        idxA.push_back(r);
                        Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();
                        
                        SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                        idxB.push_back(bInd);
                        idxB.push_back(cRow);
                        Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();

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

                        SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                        idxC.push_back(cRow);
                        idxC.push_back(r);
                        
                        auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    });
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            });
        currC = iLoopLower.getResult(0);

        if (uC > uB) {
            auto uDiff = arith::ConstantIndexOp::create(rewriter, loc, uC - uB);
            auto startUpperBand = arith::ConstantIndexOp::create(rewriter, loc, uB + 1);
            scf::ForOp iLoop = scf::ForOp::create(
                rewriter, loc, c0, uDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::AddIOp::create(ob, loc, i, startUpperBand).getResult();
                    Value nMinusBand = arith::SubIOp::create(ob, loc, totalCols, currBand).getResult();
                    Value numEls = arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult();
                    
                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cCol = arith::AddIOp::create(ib, loc, r, currBand).getResult();
                            
                            SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                            idxA.push_back(r);
                            idxA.push_back(cCol);
                            Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();
                            
                            Value operand2 = cf0; // B is outside its upper bandwidth

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

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(r);
                            idxC.push_back(cCol);
                            
                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoop.getResult(0);
        }

        auto uMinVal = std::min(uB, uC);
        if (uMinVal != 0) {
            auto totalUpperDiags = arith::ConstantIndexOp::create(rewriter, loc, uMinVal);
            scf::ForOp iLoopUpper = scf::ForOp::create(
                rewriter, loc, c0, totalUpperDiags, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter = iArgs[0];
                    Value currBand = arith::AddIOp::create(ob, loc, i, c1).getResult();
                    Value bInd = arith::AddIOp::create(ob, loc, cLB, currBand).getResult();
                    Value nMinusBand = arith::SubIOp::create(ob, loc, totalCols, currBand).getResult();
                    Value numEls = arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult();
                    
                    auto rLoop = scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner = rArgs[0];
                            Value cCol = arith::AddIOp::create(ib, loc, r, currBand).getResult();
                            
                            SmallVector<Value> idxA(batchIndices.begin(), batchIndices.end());
                            idxA.push_back(r);
                            idxA.push_back(cCol);
                            Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A, idxA).getResult();
                            
                            SmallVector<Value> idxB(batchIndices.begin(), batchIndices.end());
                            idxB.push_back(bInd);
                            idxB.push_back(r);
                            Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B, idxB).getResult();

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

                            SmallVector<Value> idxC(batchIndices.begin(), batchIndices.end());
                            idxC.push_back(r);
                            idxC.push_back(cCol);
                            
                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner, idxC);
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoopUpper.getResult(0);
        }

        return currC;
    }

    Value denseTimesDenseToDenseKernel(OpBuilder& rewriter, Location loc, 
                                       Value A, Value B, Value initC,
                                       linalg::ElementwiseKind linalgKind) const {
        
        MLIRContext* context = rewriter.getContext();
        auto inputType = cast<RankedTensorType>(A.getType());
        int64_t rank = inputType.getRank();
        
        AffineMap identityMap = rewriter.getMultiDimIdentityMap(rank);
        
        auto indexingMapsAttr = rewriter.getAffineMapArrayAttr({ identityMap, identityMap, identityMap });

        SmallVector<Attribute> iteratorTypes(
            rank, linalg::IteratorTypeAttr::get(context, utils::IteratorType::parallel));
        auto iteratorTypesAttr = rewriter.getArrayAttr(iteratorTypes);

        SmallVector<NamedAttribute> attrs{
            rewriter.getNamedAttr("kind", linalg::ElementwiseKindAttr::get(context, linalgKind)),
            rewriter.getNamedAttr("indexing_maps", indexingMapsAttr),
            rewriter.getNamedAttr("iterator_types", iteratorTypesAttr)
        };

        auto elementwiseOp = linalg::ElementwiseOp::create(
            rewriter, loc, ValueRange{ A, B }, ValueRange{ initC }, attrs);
            
        return elementwiseOp->getResult(0);
    }

public:
    using OpRewritePattern::OpRewritePattern;

    LogicalResult diaTimesDiaToDenseBandedElementwiseToSCF(
        dia::ElementwiseOp op, PatternRewriter& rewriter, const BandedSubMatrix& bandResult) const {

        Location loc{ op.getLoc() };
        Value A{ op.getInputs()[0] };
        Value B{ op.getInputs()[1] };
        Value C{ op.getOutput() };

        auto outputType{ cast<RankedTensorType>(C.getType()) };
        auto elementType{ cast<RankedTensorType>(A.getType()).getElementType() };

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

        auto rank{ outputType.getRank() };
        bool hasBatch = (rank == 3);

        auto totalRows{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 2)) };
        auto totalCols{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 1)) };

        Value currC = zeroedC;

        if (hasBatch) {
            auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
            auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
            Value batchSize = arith::ConstantIndexOp::create(rewriter, loc, outputType.getDimSize(0));

            auto bLoop = scf::ForOp::create(
                rewriter, loc, c0, batchSize, c1, ValueRange{ currC },
                [&](OpBuilder& b, Location loc, Value bIdx, ValueRange bArgs) {
                    SmallVector<Value> batchIndices = { bIdx };
                    Value loopC = diaTimesDiaToDenseKernel(
                        b, loc, op, A, B, bArgs[0], elementType,
                        totalRows, totalCols, lA, lB, uA, uB, batchIndices);
                    scf::YieldOp::create(b, loc, ValueRange{ loopC });
                });
            currC = bLoop.getResult(0);
        } else {
            currC = diaTimesDiaToDenseKernel(
                rewriter, loc, op, A, B, currC, elementType,
                totalRows, totalCols, lA, lB, uA, uB, {});
        }

        currC.getDefiningOp()->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, currC);

        return success();
    }

    LogicalResult diaTimesDiaToDiaBandedElementwiseToSCF(dia::ElementwiseOp op,
                                                         PatternRewriter& rewriter,
                                                         const BandedSubMatrix& bandResult) const {
        Location loc{ op.getLoc() };
        Value A{ op.getInputs()[0] };
        Value B{ op.getInputs()[1] };
        Value C{ op.getOutput() };

        auto resultType{ cast<RankedTensorType>(A.getType()) };
        auto elementType{ resultType.getElementType() };
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

        auto rank{ resultType.getRank() };
        bool hasBatch = (rank == 3);

        auto totalRows{ arith::ConstantIndexOp::create(
            rewriter, loc, resultType.getDimSize(rank - 1)) };

        Value currC = zeroedC;

        if (hasBatch) {
            auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
            auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
            Value batchSize = arith::ConstantIndexOp::create(rewriter, loc, resultType.getDimSize(0));

            auto bLoop = scf::ForOp::create(
                rewriter, loc, c0, batchSize, c1, ValueRange{ currC },
                [&](OpBuilder& b, Location loc, Value bIdx, ValueRange bArgs) {
                    SmallVector<Value> batchIndices = { bIdx };
                    Value loopC = diaTimesDiaToDiaKernel(
                        b, loc, op, A, B, bArgs[0], elementType,
                        totalRows, lA, lB, uA, uB, lC, batchIndices);
                    scf::YieldOp::create(b, loc, ValueRange{ loopC });
                });
            currC = bLoop.getResult(0);
        } else {
            currC = diaTimesDiaToDiaKernel(
                rewriter, loc, op, A, B, currC, elementType,
                totalRows, lA, lB, uA, uB, lC, {});
        }

        currC.getDefiningOp()->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, currC);

        return success();
    }

    LogicalResult diaTimesDenseToDiaBandedElementwiseToSCF(
        dia::ElementwiseOp op, PatternRewriter& rewriter, const BandedSubMatrix& bandResult) const {

        Location loc{ op.getLoc() };
        Value A{ op.getInputs()[0] };
        Value B{ op.getInputs()[1] };
        Value C{ op.getOutput() };

        auto elementType{ cast<RankedTensorType>(A.getType()).getElementType() };
        MLIRContext* context{ rewriter.getContext() };

        BandedSubMatrix bandA{ BandedStructureAnalysis::readPropertyFromDictAttr(
            A.getDefiningOp()->getAttrDictionary()) };

        auto lA{ bandA.Property.LowerBandwidth };
        auto uA{ bandA.Property.UpperBandwidth };
        auto lC{ bandResult.Property.LowerBandwidth };
        auto uC{ bandResult.Property.UpperBandwidth };

        Value zero{ arith::ConstantOp::create(rewriter, loc, elementType,
                                              rewriter.getZeroAttr(elementType)) };
        Value zeroedC{
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0)
        };

        auto outputType{ cast<RankedTensorType>(B.getType()) };
        auto rank{ outputType.getRank() };
        bool hasBatch = (rank == 3);

        auto totalRows{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 2)) };
        auto totalCols{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 1)) };

        Value currC = zeroedC;

        if (hasBatch) {
            auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
            auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
            Value batchSize = arith::ConstantIndexOp::create(rewriter, loc, outputType.getDimSize(0));

            auto bLoop = scf::ForOp::create(
                rewriter, loc, c0, batchSize, c1, ValueRange{ currC },
                [&](OpBuilder& b, Location loc, Value bIdx, ValueRange bArgs) {
                    SmallVector<Value> batchIndices = { bIdx };
                    Value loopC = diaTimesDenseToDiaKernel(
                        b, loc, op, A, B, bArgs[0], elementType,
                        totalRows, totalCols, lA, uA, lC, uC, batchIndices);
                    scf::YieldOp::create(b, loc, ValueRange{ loopC });
                });
            currC = bLoop.getResult(0);
        } else {
            currC = diaTimesDenseToDiaKernel(
                rewriter, loc, op, A, B, currC, elementType,
                totalRows, totalCols, lA, uA, lC, uC, {});
        }

        currC.getDefiningOp()->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, currC);
        return success();
    }

    LogicalResult denseTimesDenseToDiaBandedElementwiseToSCF(
        dia::ElementwiseOp op, PatternRewriter& rewriter, const BandedSubMatrix& bandResult) const {
        
        Location loc = op.getLoc();
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutput();
        
        auto denseType = cast<RankedTensorType>(A.getType());
        auto elementType = denseType.getElementType();
        auto rank = denseType.getRank();

        auto lC = bandResult.Property.LowerBandwidth;
        auto uC = bandResult.Property.UpperBandwidth;

        Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                               rewriter.getZeroAttr(elementType));
        Value zeroedC =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0);

        auto totalRows = arith::ConstantIndexOp::create(rewriter, loc, denseType.getDimSize(rank - 2));
        auto totalCols = arith::ConstantIndexOp::create(rewriter, loc, denseType.getDimSize(rank - 1));

        bool hasBatch = (rank == 3);
        Value currC = zeroedC;

        if (hasBatch) {
            auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
            auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
            Value batchSize = arith::ConstantIndexOp::create(rewriter, loc, denseType.getDimSize(0));

            auto bLoop = scf::ForOp::create(
                rewriter, loc, c0, batchSize, c1, ValueRange{ currC },
                [&](OpBuilder& b, Location loc, Value bIdx, ValueRange bArgs) {
                    SmallVector<Value> batchIndices = { bIdx };
                    Value loopC = denseTimesDenseToDiaKernel(
                        b, loc, op, A, B, bArgs[0], elementType,
                        totalRows, totalCols, lC, uC, batchIndices);
                    scf::YieldOp::create(b, loc, ValueRange{ loopC });
                });
            currC = bLoop.getResult(0);
        } else {
            currC = denseTimesDenseToDiaKernel(
                rewriter, loc, op, A, B, currC, elementType,
                totalRows, totalCols, lC, uC, {});
        }

        if (auto metadata = op->getAttr("metadata")) {
            currC.getDefiningOp()->setAttr("metadata", metadata);
        }

        rewriter.replaceOp(op, currC);
        return success();
    }

    LogicalResult denseTimesDiaToDiaBandedElementwiseToSCF(
        dia::ElementwiseOp op, PatternRewriter& rewriter, const BandedSubMatrix& bandResult) const {

        Location loc{ op.getLoc() };
        Value A{ op.getInputs()[0] };
        Value B{ op.getInputs()[1] };
        Value C{ op.getOutput() };

        auto outputType{ cast<RankedTensorType>(A.getType()) };
        auto elementType{ outputType.getElementType() };
        MLIRContext* context{ rewriter.getContext() };

        BandedSubMatrix bandB{ BandedStructureAnalysis::readPropertyFromDictAttr(
            B.getDefiningOp()->getAttrDictionary()) };

        auto lB{ bandB.Property.LowerBandwidth };
        auto uB{ bandB.Property.UpperBandwidth };
        auto lC{ bandResult.Property.LowerBandwidth };
        auto uC{ bandResult.Property.UpperBandwidth };

        Value zero{ arith::ConstantOp::create(rewriter, loc, elementType,
                                              rewriter.getZeroAttr(elementType)) };
        Value zeroedC{
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0)
        };

        auto rank{ outputType.getRank() };
        bool hasBatch = (rank == 3);

        auto totalRows{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 2)) };
        auto totalCols{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 1)) };

        Value currC = zeroedC;

        if (hasBatch) {
            auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
            auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
            Value batchSize = arith::ConstantIndexOp::create(rewriter, loc, outputType.getDimSize(0));

            auto bLoop = scf::ForOp::create(
                rewriter, loc, c0, batchSize, c1, ValueRange{ currC },
                [&](OpBuilder& b, Location loc, Value bIdx, ValueRange bArgs) {
                    SmallVector<Value> batchIndices = { bIdx };
                    Value loopC = denseTimesDiaToDiaKernel(
                        b, loc, op, A, B, bArgs[0], elementType,
                        totalRows, totalCols, lB, uB, lC, uC, batchIndices);
                    scf::YieldOp::create(b, loc, ValueRange{ loopC });
                });
            currC = bLoop.getResult(0);
        } else {
            currC = denseTimesDiaToDiaKernel(
                rewriter, loc, op, A, B, currC, elementType,
                totalRows, totalCols, lB, uB, lC, uC, {});
        }

        if (auto metadata = op->getAttr("metadata")) {
            currC.getDefiningOp()->setAttr("metadata", metadata);
        }

        rewriter.replaceOp(op, currC);
        return success();
    }

    LogicalResult diaToDiaBandedElementwiseToLinalg(dia::ElementwiseOp op,
                                                    PatternRewriter& rewriter,
                                                    const BandedSubMatrix& bandResult) const {
        Location loc{ op.getLoc() };
        Value input{ op.getInputs()[0] };
        Value output{ op.getOutput() };

        auto kind{ op.getKind() };
        linalg::ElementwiseKind linalgKind;

        switch (kind) {
            case (dia::ElementwiseKind::square):
                linalgKind = linalg::ElementwiseKind::square;
                break;
            default:
                return failure();
        }

        Value currC = diaToDiaKernel(rewriter, loc, input, output, linalgKind);

        if (auto metadata = op->getAttr("metadata")) {
            currC.getDefiningOp()->setAttr("metadata", metadata);
        }

        rewriter.replaceOp(op, currC);
        return success();
    }

    LogicalResult diaTimesDenseToDenseBandedElementwiseToSCF(
        dia::ElementwiseOp op, PatternRewriter& rewriter, const BandedSubMatrix& bandResult) const {

        Location loc{ op.getLoc() };
        Value A{ op.getInputs()[0] };
        Value B{ op.getInputs()[1] };
        Value C{ op.getOutput() };

        auto outputType{ cast<RankedTensorType>(C.getType()) };
        auto elementType{ cast<RankedTensorType>(A.getType()).getElementType() };
        MLIRContext* context{ rewriter.getContext() };

        BandedSubMatrix bandA{ BandedStructureAnalysis::readPropertyFromDictAttr(
            A.getDefiningOp()->getAttrDictionary()) };

        auto lA{ bandA.Property.LowerBandwidth };
        auto uA{ bandA.Property.UpperBandwidth };
        auto lC{ bandResult.Property.LowerBandwidth };
        auto uC{ bandResult.Property.UpperBandwidth };

        Value zero{ arith::ConstantOp::create(rewriter, loc, elementType,
                                              rewriter.getZeroAttr(elementType)) };
        Value zeroedC{
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0)
        };

        auto rank{ outputType.getRank() };
        bool hasBatch = (rank == 3);

        auto totalRows{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 2)) };
        auto totalCols{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 1)) };

        Value currC = zeroedC;

        if (hasBatch) {
            auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
            auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
            Value batchSize = arith::ConstantIndexOp::create(rewriter, loc, outputType.getDimSize(0));

            auto bLoop = scf::ForOp::create(
                rewriter, loc, c0, batchSize, c1, ValueRange{ currC },
                [&](OpBuilder& b, Location loc, Value bIdx, ValueRange bArgs) {
                    SmallVector<Value> batchIndices = { bIdx };
                    Value loopC = diaTimesDenseToDenseKernel(
                        b, loc, op, A, B, bArgs[0], elementType,
                        totalRows, totalCols, lA, uA, lC, uC, batchIndices);
                    scf::YieldOp::create(b, loc, ValueRange{ loopC });
                });
            currC = bLoop.getResult(0);
        } else {
            currC = diaTimesDenseToDenseKernel(
                rewriter, loc, op, A, B, currC, elementType,
                totalRows, totalCols, lA, uA, lC, uC, {});
        }

        if (auto metadata = op->getAttr("metadata")) {
            currC.getDefiningOp()->setAttr("metadata", metadata);
        }

        rewriter.replaceOp(op, currC);
        return success();
    }

    LogicalResult denseTimesDiaToDenseBandedElementwiseToSCF(
        dia::ElementwiseOp op, PatternRewriter& rewriter, const BandedSubMatrix& bandResult) const {

        Location loc{ op.getLoc() };
        Value A{ op.getInputs()[0] };
        Value B{ op.getInputs()[1] };
        Value C{ op.getOutput() };

        auto outputType{ cast<RankedTensorType>(C.getType()) };
        auto elementType{ outputType.getElementType() };
        MLIRContext* context{ rewriter.getContext() };

        BandedSubMatrix bandB{ BandedStructureAnalysis::readPropertyFromDictAttr(
            B.getDefiningOp()->getAttrDictionary()) };

        auto lB{ bandB.Property.LowerBandwidth };
        auto uB{ bandB.Property.UpperBandwidth };
        auto lC{ bandResult.Property.LowerBandwidth };
        auto uC{ bandResult.Property.UpperBandwidth };

        Value zero{ arith::ConstantOp::create(rewriter, loc, elementType,
                                              rewriter.getZeroAttr(elementType)) };
        Value zeroedC{
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0)
        };

        auto rank{ outputType.getRank() };
        bool hasBatch = (rank == 3);

        auto totalRows{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 2)) };
        auto totalCols{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 1)) };

        Value currC = zeroedC;

        if (hasBatch) {
            auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
            auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
            Value batchSize = arith::ConstantIndexOp::create(rewriter, loc, outputType.getDimSize(0));

            auto bLoop = scf::ForOp::create(
                rewriter, loc, c0, batchSize, c1, ValueRange{ currC },
                [&](OpBuilder& b, Location loc, Value bIdx, ValueRange bArgs) {
                    SmallVector<Value> batchIndices = { bIdx };
                    Value loopC = denseTimesDiaToDenseKernel(
                        b, loc, op, A, B, bArgs[0], elementType,
                        totalRows, totalCols, lB, uB, lC, uC, batchIndices);
                    scf::YieldOp::create(b, loc, ValueRange{ loopC });
                });
            currC = bLoop.getResult(0);
        } else {
            currC = denseTimesDiaToDenseKernel(
                rewriter, loc, op, A, B, currC, elementType,
                totalRows, totalCols, lB, uB, lC, uC, {});
        }

        if (auto metadata = op->getAttr("metadata")) {
            currC.getDefiningOp()->setAttr("metadata", metadata);
        }

        rewriter.replaceOp(op, currC);
        return success();
    }

    LogicalResult denseTimesDenseToDenseBandedElementwiseToLinalg(
        dia::ElementwiseOp op, PatternRewriter& rewriter, const BandedSubMatrix& bandResult) const {
        
        Location loc{ op.getLoc() };
        Value A{ op.getInputs()[0] };
        Value B{ op.getInputs()[1] };
        Value C{ op.getOutput() };

        auto kind{ op.getKind() };
        linalg::ElementwiseKind linalgKind;

        switch (kind) {
            case (dia::ElementwiseKind::mul):
                linalgKind = linalg::ElementwiseKind::mul;
                break;
            case (dia::ElementwiseKind::add):
                linalgKind = linalg::ElementwiseKind::add;
                break;
            case (dia::ElementwiseKind::sub):
                linalgKind = linalg::ElementwiseKind::sub;
                break;
            default:
                return failure();
        }

        Value currC = denseTimesDenseToDenseKernel(rewriter, loc, A, B, C, linalgKind);

        if (auto metadata = op->getAttr("metadata")) {
            currC.getDefiningOp()->setAttr("metadata", metadata);
        }

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
            else if (!bandA.IsDia && !bandB.IsDia && !opBandInfo.IsDia)
                return denseTimesDenseToDenseBandedElementwiseToLinalg(op, rewriter, opBandInfo);
            else if (!bandA.IsDia && !bandB.IsDia && opBandInfo.IsDia)
                return denseTimesDenseToDiaBandedElementwiseToSCF(op, rewriter, opBandInfo);
            else
                return failure();
        } else if (numOps == 1) {
            Value A = op.getInputs()[0];
            Value C = op.getOutput();

            Operation* defOpA = A.getDefiningOp();

            auto dictA = defOpA->getAttrDictionary();

            if (!dictA) return failure();
            bandA = BandedStructureAnalysis::readPropertyFromDictAttr(dictA);

            return diaToDiaBandedElementwiseToLinalg(op, rewriter, opBandInfo);
        } else
            return failure();
    }
};

void addDIAElementwisePatterns(RewritePatternSet& patterns) {
    patterns.add<DIAElementwisePattern>(patterns.getContext());
}

}  // namespace mlir::bpa
