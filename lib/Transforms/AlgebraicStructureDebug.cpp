#include "lib/Transforms/AlgebraicStructureDebug.h"

#include <cassert>

#include "lib/Analysis/AlgebraicStructureAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::asa {

#define GEN_PASS_DEF_ALGEBRAICSTRUCTUREDEBUG
#include "lib/Transforms/Passes.h.inc"

struct AlgebraicStructureDebugPass
    : public impl::AlgebraicStructureDebugBase<AlgebraicStructureDebugPass> {
    using AlgebraicStructureDebugBase::AlgebraicStructureDebugBase;
    void runOnOperation() {
        auto funcOp{ getOperation() };
        auto* context{ funcOp->getContext() };
        AlgebraicStructureAnalysis ASA;

        for (auto& block : funcOp.getBody()) (void)ASA.run(&block);

        Builder builder(context);

        funcOp->walk([&](Operation* inst) {
            auto results{ inst->getResults() };
            if (results.size() != 1 || !ASA.hasProperty(results[0])) return;
            auto stringAttr{ builder.getStringAttr(
                propertyToString(ASA.getProperty(results[0]).property)) };
            auto propertyAttr{ builder.getNamedAttr("algebraicProperty", stringAttr) };
            auto dim0Attr{ builder.getI64IntegerAttr(ASA.getProperty(results[0]).dimensions[0]) };
            auto dim1Attr{ builder.getI64IntegerAttr(ASA.getProperty(results[0]).dimensions[1]) };
            auto arrayAttr{ builder.getArrayAttr({ dim0Attr, dim1Attr }) };
            auto dimsAttr{ builder.getNamedAttr("propertyDims", arrayAttr) };
            auto dictAttr{ builder.getDictionaryAttr({ propertyAttr, dimsAttr }) };
            inst->setAttr("metadata", dictAttr);
        });

        return;
    }
};

}  // namespace mlir::asa
