#include "Transform/BandedPropagation.h"

#include <cstdint>

#include "Analysis/BandedStructureAnalysis.h"
#include "Utils/TransformUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::bpa {

#define GEN_PASS_DEF_BANDEDANALYSIS
#include "lib/Transform/Passes.h.inc"

struct BandedAnalysisPass : public impl::BandedAnalysisBase<BandedAnalysisPass> {
    using BandedAnalysisBase::BandedAnalysisBase;

    void runOnOperation() override {
        auto funcOp{ getOperation() };
        auto* context{ funcOp->getContext() };
        BandedStructureAnalysis BSA;

        for (auto& block : funcOp.getBody()) (void)BSA.run(&block);

        Builder builder(context);

        funcOp->walk([&](Operation* inst) {
            auto results{ inst->getResults() };
            if (results.size() != 1 || !BSA.hasProperty(results[0])) return;

            auto analysisResult{ BSA.getProperty(results[0]) };
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

            if (detectDIA && shouldCompress(analysisResult, N))
                attrs.emplace_back(builder.getNamedAttr("dia", builder.getBoolAttr(true)));

            auto dictAttr = builder.getDictionaryAttr(attrs);
            inst->setAttr("metadata", dictAttr);
        });
    }
};

}  // namespace mlir::bpa
