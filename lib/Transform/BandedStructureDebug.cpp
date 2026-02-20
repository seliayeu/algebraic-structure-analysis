#include "Transform/BandedStructureDebug.h"

#include "lib/Analysis/BandedStructureAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::bpa {

#define GEN_PASS_DEF_BANDEDSTRUCTUREDEBUG
#include "lib/Transform/Passes.h.inc"

struct BandedStructureDebugPass : public impl::BandedStructureDebugBase<BandedStructureDebugPass> {
    using BandedStructureDebugBase::BandedStructureDebugBase;
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

            auto upperAttr{ builder.getNamedAttr(
                "upperBw", builder.getI64IntegerAttr(property.UpperBandwidth)) };

            auto lowerAttr{ builder.getNamedAttr(
                "lowerBw", builder.getI64IntegerAttr(property.LowerBandwidth)) };

            auto dim0Attr{ builder.getI64IntegerAttr(dims[0]) };
            auto dim1Attr{ builder.getI64IntegerAttr(dims[1]) };
            auto dimsArrayAttr{ builder.getNamedAttr(
                "propertyDims", builder.getArrayAttr({ dim0Attr, dim1Attr })) };

            auto dictAttr{ builder.getDictionaryAttr({ upperAttr, lowerAttr, dimsArrayAttr }) };

            inst->setAttr("metadata", dictAttr);
        });
    }
};

}  // namespace mlir::bpa
