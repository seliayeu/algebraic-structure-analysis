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
    using OpRewritePattern::OpRewritePattern;

    LogicalResult diaTimesDiaToDenseBandedElementwiseToSCF(
        dia::ElementwiseOp op, PatternRewriter& rewriter, const BandedSubMatrix& bandResult) const {

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

        if (op.getKind() != dia::ElementwiseKind::mul && (lA > lB || lB > lA)) {
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
                            Value cRow{ arith::AddIOp::create(ib, loc, r, currBand) };
                            Value operand1, operand2;

                            if (lA > lB) {
                                operand1 =
                                    tensor::ExtractOp::create(ib, loc, elementType, A, { i, cRow });
                                operand2 = cf0;
                            } else {
                                operand1 = cf0;
                                operand2 =
                                    tensor::ExtractOp::create(ib, loc, elementType, B, { i, cRow });
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
                            
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ cRow, r }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
            currC = iLoop.getResult(0);
        }

        if (op.getKind() != dia::ElementwiseKind::mul && (uA > uB || uB > uA)) {
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
                            Value cCol{ arith::AddIOp::create(ib, loc, r, currBand) };
                            Value operand1, operand2;
                            
                            if (uA > uB) {
                                operand1 = tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                     { aInd, r });
                                operand2 = cf0;
                            } else {
                                operand1 = cf0;
                                operand2 = tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                     { bInd, r });
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

                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ r, cCol }) };
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
                Value aInd{ (lA > lB) ? arith::AddIOp::create(ob, loc, i, lDiff).getResult() : i };
                Value bInd{ (lA > lB) ? i : arith::AddIOp::create(ob, loc, i, lDiff).getResult() };

                Value currBand{ arith::SubIOp::create(ob, loc, lMin, i) };
                Value mMinusBand{ arith::SubIOp::create(ob, loc, totalRows, currBand) };
                Value numEls{ arith::MinUIOp::create(ob, loc, mMinusBand, totalCols) };
                auto rLoop{ scf::ForOp::create(
                    ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                    [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                        auto cInner{ rArgs[0] };
                        Value cRow{ arith::AddIOp::create(ib, loc, r, currBand) };
                        auto operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                 { aInd, cRow }) };
                        auto operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                 { bInd, cRow }) };

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
                            auto cInner{ rArgs[0] };
                            Value cCol{ arith::AddIOp::create(ib, loc, r, i) };
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
                                case dia::ElementwiseKind::mul:
                                    newOp = arith::MulFOp::create(ib, loc, operand1, operand2);
                                    break;
                                default:
                                    assert(false);
                            }

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

    LogicalResult diaTimesDiaToDiaBandedElementwiseToSCF(dia::ElementwiseOp op,
                                                         PatternRewriter& rewriter,
                                                         const BandedSubMatrix& bandResult) const {
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
        auto totalRows{ arith::ConstantIndexOp::create(
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
                        ob, loc, c0, totalRows, c1, ValueRange{ cOuter },
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
                        ob, loc, c0, totalRows, c1, ValueRange{ cOuter },
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

        auto numDiagonals{ arith::ConstantIndexOp::create(
            rewriter, loc, std::min(lA, lB) + 1 + std::min(uA, uB)) };
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
                    ob, loc, c0, totalRows, c1, ValueRange{ cOuter },
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

        auto c0{ arith::ConstantIndexOp::create(rewriter, loc, 0) };
        auto c1{ arith::ConstantIndexOp::create(rewriter, loc, 1) };
        auto cf0{ arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getFloatAttr(elementType, 0.0)) };

        auto outputType{ cast<RankedTensorType>(B.getType()) };
        auto rank{ outputType.getRank() };
        auto totalRows{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 2)) };
        auto totalCols{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 1)) };

        auto cLA{ arith::ConstantIndexOp::create(rewriter, loc, lA) };
        auto cLC{ arith::ConstantIndexOp::create(rewriter, loc, lC) };

        auto currC{ zeroedC };

        if (lC > lA) {
            auto lDiff{ arith::ConstantIndexOp::create(rewriter, loc, lC - lA) };
            scf::ForOp iLoop{ scf::ForOp::create(
                rewriter, loc, c0, lDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::SubIOp::create(ob, loc, cLC, i).getResult() };
                    Value cInd{ arith::SubIOp::create(ob, loc, cLC, currBand).getResult() };

                    Value mMinusBand{
                        arith::SubIOp::create(ob, loc, totalRows, currBand).getResult()
                    };
                    Value numEls{
                        arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult()
                    };

                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cRow{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            
                            Value operand1{ cf0 }; // A is outside its lower bandwidth
                            Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                      ValueRange{ cRow, r })
                                                .getResult() };

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
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ cInd, cRow }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
            currC = iLoop.getResult(0);
        }

        auto lMinVal{ std::min(lA, lC) };
        if (lMinVal > 0) {
            auto totalLowerDiags{ arith::ConstantIndexOp::create(rewriter, loc, lMinVal) };
            auto lMin{ arith::ConstantIndexOp::create(rewriter, loc, lMinVal) };
            scf::ForOp iLoopLower{ scf::ForOp::create(
                rewriter, loc, c0, totalLowerDiags, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::SubIOp::create(ob, loc, lMin, i).getResult() };
                    Value aInd{ arith::SubIOp::create(ob, loc, cLA, currBand).getResult() };
                    Value cInd{ arith::SubIOp::create(ob, loc, cLC, currBand).getResult() };

                    Value mMinusBand{
                        arith::SubIOp::create(ob, loc, totalRows, currBand).getResult()
                    };
                    Value numEls{
                        arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult()
                    };

                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cRow{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };

                            Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                      ValueRange{ aInd, cRow })
                                                .getResult() };
                            Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                      ValueRange{ cRow, r })
                                                .getResult() };

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
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ cInd, cRow }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
            currC = iLoopLower.getResult(0);
        }

        auto uMinVal{ std::min(uA, uC) };
        auto totalUpperDiags{ arith::ConstantIndexOp::create(rewriter, loc, uMinVal + 1) };
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

                        Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                  ValueRange{ aInd, r })
                                            .getResult() };
                        Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                  ValueRange{ r, cCol })
                                            .getResult() };

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
                        auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                               ValueRange{ cInd, r }) };
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    }) };
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            }) };
        currC = iLoopUpper.getResult(0);

        if (uC > uA) {
            auto uDiff{ arith::ConstantIndexOp::create(rewriter, loc, uC - uA) };
            auto startUpperBand{ arith::ConstantIndexOp::create(rewriter, loc, uA + 1) };
            scf::ForOp iLoop{ scf::ForOp::create(
                rewriter, loc, c0, uDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::AddIOp::create(ob, loc, i, startUpperBand).getResult() };
                    Value cInd{ arith::AddIOp::create(ob, loc, cLC, currBand).getResult() };

                    Value nMinusBand{
                        arith::SubIOp::create(ob, loc, totalCols, currBand).getResult()
                    };
                    Value numEls{
                        arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult()
                    };

                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cCol{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            
                            Value operand1{ cf0 }; // A is outside its upper bandwidth
                            Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                      ValueRange{ r, cCol })
                                                .getResult() };

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
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ cInd, r }) };
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

    LogicalResult denseTimesDenseToDiaBandedElementwiseToSCF(
        dia::ElementwiseOp op, PatternRewriter& rewriter, const BandedSubMatrix& bandResult) const {
        
        Location loc = op.getLoc();
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutput();
        
        auto elementType = cast<RankedTensorType>(A.getType()).getElementType();
        auto denseType = cast<RankedTensorType>(A.getType());
        auto rank = denseType.getRank();

        auto lC = bandResult.Property.LowerBandwidth;
        auto uC = bandResult.Property.UpperBandwidth;

        Value zero = arith::ConstantOp::create(rewriter, loc, elementType,
                                               rewriter.getZeroAttr(elementType));
        Value zeroedC =
            linalg::FillOp::create(rewriter, loc, ValueRange{ zero }, ValueRange{ C }).getResult(0);

        auto c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
        auto c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

        auto totalRows = arith::ConstantIndexOp::create(rewriter, loc,
                                                        denseType.getDimSize(rank - 2));
        auto totalCols = arith::ConstantIndexOp::create(rewriter, loc,
                                                        denseType.getDimSize(rank - 1));
        auto cLC = arith::ConstantIndexOp::create(rewriter, loc, lC);

        auto currC = zeroedC;

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

                        Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                   ValueRange{ cRow, r })
                                             .getResult();
                        Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                   ValueRange{ cRow, r })
                                             .getResult();

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
                        auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                ValueRange{ cInd, cRow });
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

                            Value operand1 = tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                       ValueRange{ r, cCol })
                                                 .getResult();
                            Value operand2 = tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                       ValueRange{ r, cCol })
                                                 .getResult();

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
                            auto updated = tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                    ValueRange{ cInd, r });
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        });
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                });
            currC = iLoopUpper.getResult(0);
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

        auto elementType{ cast<RankedTensorType>(A.getType()).getElementType() };
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

        auto c0{ arith::ConstantIndexOp::create(rewriter, loc, 0) };
        auto c1{ arith::ConstantIndexOp::create(rewriter, loc, 1) };
        auto cf0{ arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getFloatAttr(elementType, 0.0)) };

        auto outputType{ cast<RankedTensorType>(A.getType()) };
        auto rank{ outputType.getRank() };
        auto totalRows{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 2)) };
        auto totalCols{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 1)) };

        auto cLB{ arith::ConstantIndexOp::create(rewriter, loc, lB) };
        auto cLC{ arith::ConstantIndexOp::create(rewriter, loc, lC) };

        auto currC{ zeroedC };

        if (lC > lB) {
            auto lDiff{ arith::ConstantIndexOp::create(rewriter, loc, lC - lB) };
            scf::ForOp iLoop{ scf::ForOp::create(
                rewriter, loc, c0, lDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::SubIOp::create(ob, loc, cLC, i).getResult() };
                    Value cInd{ arith::SubIOp::create(ob, loc, cLC, currBand).getResult() };

                    Value mMinusBand{
                        arith::SubIOp::create(ob, loc, totalRows, currBand).getResult()
                    };
                    Value numEls{
                        arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult()
                    };

                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cRow{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            
                            Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                      ValueRange{ cRow, r })
                                                .getResult() };
                            Value operand2{ cf0 }; // B is outside its lower bandwidth

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
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ cInd, cRow }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
            currC = iLoop.getResult(0);
        }

        // 2. Lower bands shared by B and C
        auto lMinVal{ std::min(lB, lC) };
        if (lMinVal > 0) {
            auto totalLowerDiags{ arith::ConstantIndexOp::create(rewriter, loc, lMinVal) };
            auto lMin{ arith::ConstantIndexOp::create(rewriter, loc, lMinVal) };
            scf::ForOp iLoopLower{ scf::ForOp::create(
                rewriter, loc, c0, totalLowerDiags, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::SubIOp::create(ob, loc, lMin, i).getResult() };
                    Value bInd{ arith::SubIOp::create(ob, loc, cLB, currBand).getResult() };
                    Value cInd{ arith::SubIOp::create(ob, loc, cLC, currBand).getResult() };

                    Value mMinusBand{
                        arith::SubIOp::create(ob, loc, totalRows, currBand).getResult()
                    };
                    Value numEls{
                        arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult()
                    };

                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cRow{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };

                            Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                      ValueRange{ cRow, r })
                                                .getResult() };
                            Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                      ValueRange{ bInd, cRow })
                                                .getResult() };

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
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ cInd, cRow }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
            currC = iLoopLower.getResult(0);
        }

        // 3. Main diagonal and upper bands shared by B and C
        auto uMinVal{ std::min(uB, uC) };
        auto totalUpperDiags{ arith::ConstantIndexOp::create(rewriter, loc, uMinVal + 1) };
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

                        Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                  ValueRange{ r, cCol })
                                            .getResult() };
                        Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                  ValueRange{ bInd, r })
                                            .getResult() };

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
                        auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                               ValueRange{ cInd, r }) };
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    }) };
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            }) };
        currC = iLoopUpper.getResult(0);

        // 4. Upper bands present in C but not in B
        if (uC > uB) {
            auto uDiff{ arith::ConstantIndexOp::create(rewriter, loc, uC - uB) };
            auto startUpperBand{ arith::ConstantIndexOp::create(rewriter, loc, uB + 1) };
            scf::ForOp iLoop{ scf::ForOp::create(
                rewriter, loc, c0, uDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::AddIOp::create(ob, loc, i, startUpperBand).getResult() };
                    Value cInd{ arith::AddIOp::create(ob, loc, cLC, currBand).getResult() };

                    Value nMinusBand{
                        arith::SubIOp::create(ob, loc, totalCols, currBand).getResult()
                    };
                    Value numEls{
                        arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult()
                    };

                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cCol{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            
                            Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                      ValueRange{ r, cCol })
                                                .getResult() };
                            Value operand2{ cf0 }; // B is outside its upper bandwidth

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
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ cInd, r }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
            currC = iLoop.getResult(0);
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

        auto c0{ arith::ConstantIndexOp::create(rewriter, loc, 0) };
        auto c1{ arith::ConstantIndexOp::create(rewriter, loc, 1) };
        auto cf0{ arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getFloatAttr(elementType, 0.0)) };

        auto rank{ outputType.getRank() };
        auto totalRows{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 2)) };
        auto totalCols{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 1)) };

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
                    Value mMinusBand{
                        arith::SubIOp::create(ob, loc, totalRows, currBand).getResult()
                    };
                    Value numEls{
                        arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult()
                    };
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cRow{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            Value operand1{ cf0 };
                            Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                      ValueRange{ cRow, r })
                                                .getResult() };
                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2)
                                                .getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2)
                                                .getResult();
                                    break;
                                default:
                                    assert(false);
                            }
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ cRow, r }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
            currC = iLoop.getResult(0);
        }

        auto lMinVal{ std::min(lA, lC) };
        auto totalLowerDiags{ arith::ConstantIndexOp::create(rewriter, loc,
                                                             lMinVal + 1) };  // do main diag
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
                        Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                  ValueRange{ aInd, cRow })
                                            .getResult() };
                        Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                  ValueRange{ cRow, r })
                                            .getResult() };
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
                            default:
                                assert(false);
                        }
                        auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                               ValueRange{ cRow, r }) };
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    }) };
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            }) };
        currC = iLoopLower.getResult(0);

        if (uC > uA) {
            auto uDiff{ arith::ConstantIndexOp::create(rewriter, loc, uC - uA) };
            auto startUpperBand{ arith::ConstantIndexOp::create(rewriter, loc, uA + 1) };
            scf::ForOp iLoop{ scf::ForOp::create(
                rewriter, loc, c0, uDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::AddIOp::create(ob, loc, i, startUpperBand).getResult() };
                    Value nMinusBand{
                        arith::SubIOp::create(ob, loc, totalCols, currBand).getResult()
                    };
                    Value numEls{
                        arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult()
                    };
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cCol{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            Value operand1{ cf0 };
                            Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                      ValueRange{ r, cCol })
                                                .getResult() };
                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2)
                                                .getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2)
                                                .getResult();
                                    break;
                                default:
                                    assert(false);
                            }
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ r, cCol }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
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
                    Value nMinusBand{
                        arith::SubIOp::create(ob, loc, totalCols, currBand).getResult()
                    };
                    Value numEls{
                        arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult()
                    };
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cCol{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                      ValueRange{ aInd, r })
                                                .getResult() };
                            Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                      ValueRange{ r, cCol })
                                                .getResult() };
                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2)
                                                .getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2)
                                                .getResult();
                                    break;
                                default:
                                    assert(false);
                            }
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ r, cCol }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
            currC = iLoopUpper.getResult(0);
        }

        currC.getDefiningOp()->setAttr("metadata", op->getAttr("metadata"));
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
        auto elementType{ cast<RankedTensorType>(A.getType()).getElementType() };
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

        auto c0{ arith::ConstantIndexOp::create(rewriter, loc, 0) };
        auto c1{ arith::ConstantIndexOp::create(rewriter, loc, 1) };
        auto cf0{ arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getFloatAttr(elementType, 0.0)) };

        auto rank{ outputType.getRank() };
        auto totalRows{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 2)) };
        auto totalCols{ arith::ConstantIndexOp::create(rewriter, loc,
                                                       outputType.getDimSize(rank - 1)) };

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
                    Value mMinusBand{
                        arith::SubIOp::create(ob, loc, totalRows, currBand).getResult()
                    };
                    Value numEls{
                        arith::MinUIOp::create(ob, loc, mMinusBand, totalCols).getResult()
                    };
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cRow{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                      ValueRange{ cRow, r })
                                                .getResult() };
                            Value operand2{ cf0 };
                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2)
                                                .getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2)
                                                .getResult();
                                    break;
                                default:
                                    assert(false);
                            }
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ cRow, r }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
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
                        Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                  ValueRange{ cRow, r })
                                            .getResult() };
                        Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                  ValueRange{ bInd, cRow })
                                            .getResult() };
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
                            default:
                                assert(false);
                        }
                        auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                               ValueRange{ cRow, r }) };
                        scf::YieldOp::create(ib, loc, ValueRange{ updated });
                    }) };
                scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
            }) };
        currC = iLoopLower.getResult(0);

        if (uC > uB) {
            auto uDiff{ arith::ConstantIndexOp::create(rewriter, loc, uC - uB) };
            auto startUpperBand{ arith::ConstantIndexOp::create(rewriter, loc, uB + 1) };
            scf::ForOp iLoop{ scf::ForOp::create(
                rewriter, loc, c0, uDiff, c1, ValueRange{ currC },
                [&](OpBuilder& ob, Location loc, Value i, ValueRange iArgs) {
                    auto cOuter{ iArgs[0] };
                    Value currBand{ arith::AddIOp::create(ob, loc, i, startUpperBand).getResult() };
                    Value nMinusBand{
                        arith::SubIOp::create(ob, loc, totalCols, currBand).getResult()
                    };
                    Value numEls{
                        arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult()
                    };
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cCol{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                      ValueRange{ r, cCol })
                                                .getResult() };
                            Value operand2{ cf0 };
                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2)
                                                .getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2)
                                                .getResult();
                                    break;
                                default:
                                    assert(false);
                            }
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ r, cCol }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
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
                    Value nMinusBand{
                        arith::SubIOp::create(ob, loc, totalCols, currBand).getResult()
                    };
                    Value numEls{
                        arith::MinUIOp::create(ob, loc, totalRows, nMinusBand).getResult()
                    };
                    auto rLoop{ scf::ForOp::create(
                        ob, loc, c0, numEls, c1, ValueRange{ cOuter },
                        [&](OpBuilder& ib, Location loc, Value r, ValueRange rArgs) {
                            auto cInner{ rArgs[0] };
                            Value cCol{ arith::AddIOp::create(ib, loc, r, currBand).getResult() };
                            Value operand1{ tensor::ExtractOp::create(ib, loc, elementType, A,
                                                                      ValueRange{ r, cCol })
                                                .getResult() };
                            Value operand2{ tensor::ExtractOp::create(ib, loc, elementType, B,
                                                                      ValueRange{ bInd, r })
                                                .getResult() };
                            Value newOp;
                            switch (op.getKind()) {
                                case dia::ElementwiseKind::add:
                                    newOp = arith::AddFOp::create(ib, loc, operand1, operand2)
                                                .getResult();
                                    break;
                                case dia::ElementwiseKind::sub:
                                    newOp = arith::SubFOp::create(ib, loc, operand1, operand2)
                                                .getResult();
                                    break;
                                default:
                                    assert(false);
                            }
                            auto updated{ tensor::InsertOp::create(ib, loc, newOp, cInner,
                                                                   ValueRange{ r, cCol }) };
                            scf::YieldOp::create(ib, loc, ValueRange{ updated });
                        }) };
                    scf::YieldOp::create(ob, loc, ValueRange{ rLoop.getResult(0) });
                }) };
            currC = iLoopUpper.getResult(0);
        }

        currC.getDefiningOp()->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, currC);
        return success();
    }

    LogicalResult denseTimesDenseToDenseBandedElementwiseToLinalg(dia::ElementwiseOp op, PatternRewriter& rewriter, const BandedSubMatrix& bandResult) const {
        Location loc{ op.getLoc() };
        Value A{ op.getInputs()[0] };
        Value B{ op.getInputs()[1] };
        Value C{ op.getOutput() };

        Operation* defOpA = A.getDefiningOp();
        Operation* defOpB = B.getDefiningOp();

        auto dictA = defOpA->getAttrDictionary();
        auto dictB = defOpB->getAttrDictionary();

        auto bandA = BandedStructureAnalysis::readPropertyFromDictAttr(dictA);
        auto bandB = BandedStructureAnalysis::readPropertyFromDictAttr(dictB);

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

        MLIRContext* context{ rewriter.getContext() };

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

        auto inputType{ cast<RankedTensorType>(A.getType()) };
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

        auto newOp{ linalg::ElementwiseOp::create(rewriter, loc, ValueRange{ A, B },
                                                          ValueRange{ C }, attrs) };

        if (auto metadata = op->getAttr("metadata")) newOp->setAttr("metadata", metadata);
        rewriter.replaceOp(op, newOp);
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
