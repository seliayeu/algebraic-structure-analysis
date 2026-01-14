#include "lib/Transforms/AlgebraicStructureRewrite.h"
#include "lib/Analysis/AlegbraicStructureAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"

namespace mlir::asa {

#define GEN_PASS_DEF_ALGEBRAICSTRUCTUREREWRITE
#include "lib/Transforms/AlgebraicStructureRewrite.h.inc"

struct AlgebraicStructureRewrite : public impl::AlgebraicStructureRewriteBase<AlgebraicStructureRewrite> {
    using AlgebraicStructureRewriteBase::AlgebraicStructureRewriteBase;
    void runOnOperation() {
        auto* op{ getOperation() };       
        DataFlowSolver solver{};
        solver.load<dataflow::DeadCodeAnalysis>();
        solver.load<AlgebraicStructureAnalysis>();
        if (failed(solver.initializeAndRun(op)))
            return signalPassFailure();
        return;
    }
};

}
