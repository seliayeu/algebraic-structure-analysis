#include "Transform/BandedRewrite.h"

#include <cstdint>

#include "Analysis/BandedStructureAnalysis.h"
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

struct BandedMatmulToGenericPattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern::OpRewritePattern;

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

    LogicalResult bandedSCFRewrite(linalg::MatmulOp op, PatternRewriter& rewriter) const {
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

        // llvm::outs() << dyn_cast<uint64_t>(lowerA) << " \n";

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

    LogicalResult bandedRewrite(linalg::MatmulOp op, PatternRewriter& rewriter,
                                BandedSubMatrix& opBandInfo) const {
        auto loc = op.getLoc();
        Value A = op.getInputs()[0];
        Value B = op.getInputs()[1];
        Value C = op.getOutputs()[0];
        MLIRContext* context = rewriter.getContext();
        // C is the projection of the Minkowski sum of A and B bands.

        AffineExpr d0 = rewriter.getAffineDimExpr(0);
        AffineExpr d1 = rewriter.getAffineDimExpr(1);
        AffineExpr d2 = rewriter.getAffineDimExpr(2);

        AffineMap mapA = AffineMap::get(3, 0, { d0, d2 }, context);
        AffineMap mapB = AffineMap::get(3, 0, { d2, d1 }, context);
        AffineMap mapC = AffineMap::get(3, 0, { d0, d1 }, context);

        SmallVector<AffineMap> indexingMaps = { mapA, mapB, mapC };

        SmallVector<utils::IteratorType> iteratorTypes = { utils::IteratorType::parallel,
                                                           utils::IteratorType::parallel,
                                                           utils::IteratorType::reduction };

        auto genericOp = linalg::GenericOp::create(
            rewriter, loc, TypeRange{ op.getResult(0).getType() }, ValueRange{ A, B },
            ValueRange{ C }, indexingMaps, iteratorTypes,
            [&](OpBuilder& b, Location loc, ValueRange args) {
                Value mul = arith::MulFOp::create(b, loc, args[0], args[1]);
                Value add = arith::AddFOp::create(b, loc, args[2], mul);
                linalg::YieldOp::create(b, loc, ValueRange{ add });
            });

        genericOp->setAttr("metadata", op->getAttr("metadata"));
        rewriter.replaceOp(op, genericOp.getResult(0));
        return success();
    }

    LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter& rewriter) const override {
        auto dict = op->getAttrDictionary();

        if (!dict) dict = DictionaryAttr();

        BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        auto lower = opBandInfo.Property.LowerBandwidth;
        auto upper = opBandInfo.Property.UpperBandwidth;
        auto resultType = cast<RankedTensorType>(op.getResult(0).getType());
        uint64_t n = resultType.getDimSize(0);

        if (lower == 0 && upper == 0)
            return diagRewrite(op, rewriter);
        else if (lower < n && upper < n)
            return bandedSCFRewrite(op, rewriter);
        // return bandedRewrite(op, rewriter, opBandInfo);
        return failure();
    }
};

struct BandedRewrite : public impl::BandedRewriteBase<BandedRewrite> {
    using BandedRewriteBase::BandedRewriteBase;

    void runOnOperation() override {
        func::FuncOp funcOp = getOperation();
        MLIRContext* context = funcOp.getContext();

        RewritePatternSet patterns(context);

        patterns.add<BandedMatmulToGenericPattern>(context);

        GreedyRewriteConfig config;
        config.setMaxIterations(1);
        config.setUseTopDownTraversal(true);

        (void)applyPatternsGreedily(funcOp, std::move(patterns), config);
    }
};
}  // namespace mlir::bpa
