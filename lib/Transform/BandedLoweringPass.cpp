#include "Transform/BandedLoweringPass.h"

#include "lib/Analysis/BandedStructureAnalysis.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bpa {

struct DiagonalMatmulToGenericPattern : public OpRewritePattern<linalg::MatmulOp> {
    using OpRewritePattern::OpRewritePattern;

    LogicalResult matchAndRewrite(linalg::MatmulOp op, PatternRewriter& rewriter) const override {
        auto dict = op->getAttrDictionary();

        if (!dict) dict = DictionaryAttr();

        BandedSubMatrix opBandInfo = BandedStructureAnalysis::readPropertyFromDictAttr(dict);

        if (opBandInfo.Property.LowerBandwidth != 0 && opBandInfo.Property.UpperBandwidth != 0)
            return success();

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
};

void BandedLoweringPass::runOnOperation() {
    func::FuncOp funcOp = getOperation();
    MLIRContext* context = funcOp.getContext();

    RewritePatternSet patterns(context);

    patterns.add<DiagonalMatmulToGenericPattern>(context);

    GreedyRewriteConfig config;
    config.setMaxIterations(1);
    config.setUseTopDownTraversal(true);

    (void)applyPatternsGreedily(funcOp, std::move(patterns), config);
}

void registerBandedLoweringPass() {
    PassRegistration<BandedLoweringPass>();
}

}  // namespace mlir::bpa
