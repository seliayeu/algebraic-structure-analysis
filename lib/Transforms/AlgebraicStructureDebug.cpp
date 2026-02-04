#include <cassert>
#include "lib/Transforms/AlgebraicStructureDebug.h"
#include "lib/Analysis/AlgebraicStructureAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::asa {

#define GEN_PASS_DEF_ALGEBRAICSTRUCTUREDEBUG
#include "lib/Transforms/Passes.h.inc"

struct AlgebraicStructureDebugPass : public impl::AlgebraicStructureDebugBase<AlgebraicStructureDebugPass> {
    using AlgebraicStructureDebugBase::AlgebraicStructureDebugBase;
    void runOnOperation() {
        auto funcOp{ getOperation() };       
        auto* context{ funcOp->getContext() };
        AlgebraicStructureAnalysis ASA;

        for (auto& block : funcOp.getBody())
            (void) ASA.run(&block);

        Builder builder(context);

        funcOp->walk([&](Operation* inst) {
            auto results{ inst->getResults() };
            if (results.size() != 1 || !ASA.hasProperty(results[0]))
                return;
            auto stringAttr{ builder.getStringAttr(propertyAsStringRef(ASA.getProperty(results[0]))) };
            auto propertyAttr{ builder.getNamedAttr("algebraicProperty", stringAttr) };
            auto dictAttr{ builder.getDictionaryAttr({ propertyAttr }) };
            inst->setAttr("metadata", dictAttr);
        });

        return;
    }
};

}
