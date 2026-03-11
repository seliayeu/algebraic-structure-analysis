#include "Dialect/DIA/DIAOps.h"

#include <cstdint>

#include "Dialect/DIA/DIAOps.h"
#include "Utils/TransformUtils.h"

#define GET_OP_CLASSES
#include "Dialect/DIA/DIAOps.cpp.inc"

using namespace mlir;
using namespace mlir::bpa;
using namespace mlir::bpa::dia;

struct FromDensePattern : public mlir::OpRewritePattern<FromDenseOp> {
    using OpRewritePattern::OpRewritePattern;

    llvm::LogicalResult matchAndRewrite(FromDenseOp op,
                                        mlir::PatternRewriter& rewriter) const override {
        auto constOp = op.getInput().getDefiningOp<arith::ConstantOp>();
        if (!constOp) return failure();

        auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
        if (!denseAttr) return failure();

        auto metadataAttr = op->getAttrOfType<DictionaryAttr>("metadata");
        if (!metadataAttr) return failure();

        const uint64_t upperBw = cast<IntegerAttr>(metadataAttr.get("upperBw")).getInt();
        const uint64_t lowerBw = cast<IntegerAttr>(metadataAttr.get("lowerBw")).getInt();

        auto inputType = cast<RankedTensorType>(denseAttr.getType());

        const uint64_t M = inputType.getDimSize(0);
        const int64_t N = inputType.getDimSize(1);

        const int64_t numDiags = upperBw + lowerBw + 1;

        SmallVector<float> diaValues;
        auto values = denseAttr.getValues<float>();

        for (int64_t d = 0; d < numDiags; ++d) {
            int64_t k = -lowerBw + d;
            for (int64_t j = 0; j < N; ++j) {
                int64_t i = j - k;
                if (i < 0 || i >= M) {
                    diaValues.push_back(0.0f);
                } else {
                    diaValues.push_back(
                        values[{ static_cast<uint64_t>(i), static_cast<uint64_t>(j) }]);
                }
            }
        }
        auto outputType = RankedTensorType::get({ numDiags, N }, inputType.getElementType());
        auto denseValues = DenseElementsAttr::get(outputType, ArrayRef<float>(diaValues));

        auto diaConst = arith::ConstantOp::create(rewriter, op->getLoc(), outputType, denseValues);

        diaConst->setAttr("metadata", metadataAttr);

        rewriter.replaceOp(op, diaConst);
        return success();
    }
};

void FromDenseOp::getCanonicalizationPatterns(RewritePatternSet& patterns, MLIRContext* ctx) {
    patterns.add<FromDensePattern>(ctx);
}
