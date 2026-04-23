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
            int64_t k = -static_cast<int64_t>(lowerBw) + d;
            for (int64_t i = 0; i < static_cast<int64_t>(M); ++i) {
                int64_t j = i + k;
                if (j < 0 || j >= N) {
                    diaValues.push_back(0.0f);
                } else {
                    diaValues.push_back(
                        values[{ static_cast<uint64_t>(i), static_cast<uint64_t>(j) }]);
                }
            }
        }
        auto outputType = RankedTensorType::get({ numDiags, static_cast<int64_t>(M) },
                                                inputType.getElementType());
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

LogicalResult dia::MatmulOp::verify() {
    auto lhsType = cast<RankedTensorType>(getLhs().getType());
    auto rhsType = cast<RankedTensorType>(getRhs().getType());

    if (lhsType.getDimSize(1) != rhsType.getDimSize(1))
        return emitOpError("number of columns must match: lhs has ")
               << lhsType.getDimSize(1) << " but rhs has " << rhsType.getDimSize(1);
    if (lhsType.isDynamicDim(1)) return emitOpError("lhs column dimension must be static");
    if (rhsType.isDynamicDim(1)) return emitOpError("rhs column dimension must be static");

    return success();
}

LogicalResult dia::BatchMatmulOp::verify() {
    auto lhsType = cast<RankedTensorType>(getLhs().getType());
    auto rhsType = cast<RankedTensorType>(getRhs().getType());

    if (lhsType.getDimSize(2) != rhsType.getDimSize(2))
        return emitOpError("number of columns must match: lhs has ")
               << lhsType.getDimSize(2) << " but rhs has " << rhsType.getDimSize(2);
    if (lhsType.isDynamicDim(2)) return emitOpError("lhs column dimension must be static");
    if (rhsType.isDynamicDim(2)) return emitOpError("rhs column dimension must be static");

    return success();
}

LogicalResult dia::ElementwiseOp::verify() {
    auto op1Type{ cast<RankedTensorType>(getInputs()[0].getType()) };
    auto outputType{ cast<RankedTensorType>(getOutput().getType()) };
    auto rank{ outputType.getRank() };
    if (op1Type.isDynamicDim(rank - 1) || outputType.isDynamicDim(rank - 1))
        return emitOpError("first operand column dimension must be static");
    if (getOperands().size() == 2) return success();

    auto op2Type{ cast<RankedTensorType>(getInputs()[1].getType()) };
    if (op2Type.isDynamicDim(rank - 1))
        return emitOpError("second operand column dimension must be static");

    if (op1Type.getDimSize(rank - 1) != op2Type.getDimSize(rank - 1))
        return emitOpError("number of columns must match: first operand has ")
               << op1Type.getDimSize(rank - 1) << " but second oeprand has "
               << op2Type.getDimSize(rank - 1);
    return success();
}
