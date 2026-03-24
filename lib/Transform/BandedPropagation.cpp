#include "Transform/BandedPropagation.h"

#include <cstdint>
#include <iostream>

#include "Analysis/BandedStructureAnalysis.h"
#include "Utils/TransformUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::bpa {

#define GEN_PASS_DEF_BANDEDANALYSIS
#include "lib/Transform/Passes.h.inc"

struct BandedAnalysisPass : public impl::BandedAnalysisBase<BandedAnalysisPass> {
    using BandedAnalysisBase::BandedAnalysisBase;

    LogicalResult updateShape(TypedValue<RankedTensorType> result, int64_t N,
                              BandedStructureAnalysis& BSA) {
        if (!BSA.hasProperty(result)) return failure();

        BandedSubMatrix analysisResult{ BSA.getProperty(result) };

        if (!analysisResult.IsDia) return failure();

        auto resultType = dyn_cast<RankedTensorType>(result.getType());
        if (!resultType) return failure();

        int64_t lC = static_cast<int64_t>(analysisResult.Property.UpperBandwidth);
        int64_t uC = static_cast<int64_t>(analysisResult.Property.LowerBandwidth);
        int64_t numDiags = std::min(2 * N - 1, lC + uC + 1);

        auto newType = RankedTensorType::get({ numDiags, N }, resultType.getElementType());
        result.setType(newType);
        return success();
    }

    void runOnOperation() override {
        auto funcOp{ getOperation() };
        auto* context{ funcOp->getContext() };
        BandedStructureAnalysis BSA(detectDIA);

        for (auto& block : funcOp.getBody()) (void)BSA.run(&block);

        Builder builder(context);

        funcOp->walk([&](Operation* inst) {
            auto results{ inst->getResults() };
            if (results.size() != 1 || !BSA.hasProperty(results[0])) return;

            BandedSubMatrix analysisResult{ BSA.getProperty(results[0]) };
            auto property{ analysisResult.Property };
            auto dims{ analysisResult.Dims };

            auto resultType = dyn_cast<RankedTensorType>(results[0].getType());
            if (!resultType) return;

            const uint64_t N = resultType.getDimSize(1);

            auto upperAttr{ builder.getNamedAttr(
                "upperBw", builder.getI64IntegerAttr(property.UpperBandwidth)) };

            auto lowerAttr{ builder.getNamedAttr(
                "lowerBw", builder.getI64IntegerAttr(property.LowerBandwidth)) };

            auto dim0Attr{ builder.getI64IntegerAttr(dims[0]) };
            auto dim1Attr{ builder.getI64IntegerAttr(dims[1]) };
            auto dimsArrayAttr{ builder.getNamedAttr(
                "propertyDims", builder.getArrayAttr({ dim0Attr, dim1Attr })) };

            llvm::SmallVector<mlir::NamedAttribute> attrs{ upperAttr, lowerAttr, dimsArrayAttr };

            if (detectDIA) {
                if (shouldCompressResult(*inst, analysisResult, N)) {
                    attrs.emplace_back(builder.getNamedAttr("dia", builder.getBoolAttr(true)));
                    analysisResult.IsDia = true;
                }
            } else if (analysisResult.IsDia)
                attrs.emplace_back(builder.getNamedAttr("dia", builder.getBoolAttr(true)));

            auto dictAttr = builder.getDictionaryAttr(attrs);
            inst->setAttr("metadata", dictAttr);
        });

        funcOp->walk([&](Operation* inst) {
            if (!dyn_cast<dia::MatmulOp>(inst) && !dyn_cast<dia::ElementwiseOp>(inst)) return;
            auto result{ inst->getResult(0) };
            if (cast<RankedTensorType>(result.getType()).hasStaticShape()) return;

            // works because assumed square
            int64_t N = cast<RankedTensorType>(inst->getOperand(0).getType()).getDimSize(1);

            Value output;
            if (auto matmulOp{ dyn_cast<dia::MatmulOp>(inst) }) {
                output = matmulOp.getOutput();
                if (failed(updateShape(matmulOp.getResult(), N, BSA))) return;
            } else if (auto elementwiseOp{ dyn_cast<dia::ElementwiseOp>(inst) }) {
                output = elementwiseOp.getOutput();
                if (failed(updateShape(elementwiseOp.getResult(), N, BSA))) return;
            } else {
                return;
            }

            output.setType(cast<RankedTensorType>(result.getType()));
            if (auto emptyOp = output.getDefiningOp<tensor::EmptyOp>()) {
                emptyOp.getResult().setType(cast<RankedTensorType>(result.getType()));
                emptyOp->setOperands({});
            }
        });

        auto& result = getAnalysis<BandedAnalysisResult>();
        result.detectDIA = detectDIA;
        markAnalysesPreserved<BandedAnalysisResult>();
    }
};

}  // namespace mlir::bpa
