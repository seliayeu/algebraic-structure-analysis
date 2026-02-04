#include <cassert>
#include "lib/Transforms/AlgebraicStructureDebug.h"
#include "lib/Analysis/AlegbraicStructureAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::asa {

#define GEN_PASS_DEF_ALGEBRAICSTRUCTUREDEBUG
#include "lib/Transforms/Passes.h.inc"

struct AlgebraicStructureDebugPass : public impl::AlgebraicStructureDebugBase<AlgebraicStructureDebugPass> {
    using AlgebraicStructureDebugBase::AlgebraicStructureDebugBase;
    void runOnOperation() {
        auto* op{ getOperation() };       

        auto* context{ op->getContext() };
        DataFlowSolver solver{};
        solver.load<dataflow::DeadCodeAnalysis>();
        solver.load<dataflow::SparseConstantPropagation>();
        solver.load<AlgebraicStructureAnalysis>();

        if (failed(solver.initializeAndRun(op)))
            return signalPassFailure();
    
        op->walk([&](Operation *inst) {
            for (Value result : inst->getResults()) {
                const auto *lattice{ solver.lookupState<AlgebraicStructureAnalysis::AlgebraicStructureAnalysisLattice>(result) };
                if (lattice) {
                    auto stateStr{ lattice->getValue().toString() }; 
                    NamedAttribute namedAttr("analysisState", StringAttr::get(inst->getContext(), stateStr));
                    inst->setAttr("metadata", DictionaryAttr::get(inst->getContext(), { namedAttr }));
                }
            }
        });

        return;
    }
};

}
