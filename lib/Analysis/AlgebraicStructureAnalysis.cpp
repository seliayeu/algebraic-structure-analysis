#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "lib/Analysis/AlegbraicStructureAnalysis.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "llvm/Support/DebugLog.h"
#include <iostream>

namespace mlir::asa {

// LogicalResult AlgebraicStructureAnalysis::initialize(Operation* top) {
//     for (Region &region : top->getRegions()) {
//         if (region.empty())
//             continue;
//         for (Block &block : region.getBlocks()) {
//             if (block.empty())
//                 continue;
//             for (Operation &op : block.getOperations()) {
//                 if (failed(initializeOperation(&op)))
//                     return failure();
//             }
//         }
//     }
//
//     return success();
// }

// LogicalResult AlgebraicStructureAnalysis::initializeOperation(Operation* op) {
//     if (auto metadata{ op->getAttrOfType<DictionaryAttr>("metadata") }) {
//         auto strAttr{ metadata.getAs<StringAttr>("analysisState") };
//         auto strValue{ strAttr.getValue() };
//         auto initStateValue{ AlgebraicStructureAnalysisState::ASAValue::Unknown };
//         if (strValue.data() != nullptr)
//             initStateValue = AlgebraicStructureAnalysisState::stringAsValue(strValue);
//         (void)getOrCreate<AlgebraicStructureAnalysisState>(getProgramPointAfter(op))->setValue(initStateValue);
//
//     } else {
//         (void)getOrCreate<AlgebraicStructureAnalysisState>(getProgramPointAfter(op))->setValue(AlgebraicStructureAnalysisState::ASAValue::Unknown);
//     }
//
//     if (op->getNumRegions() && failed(initialize(op)))
//         return failure();
//
//     if (failed(visit(getProgramPointAfter(op))))
//         return failure();
//
//     return success();
// }

LogicalResult AlgebraicStructureAnalysis::visitOperation(Operation *op,
                                       ArrayRef<const AlgebraicStructureAnalysisLattice*> operands,
                                       ArrayRef<AlgebraicStructureAnalysisLattice*> results) {
    if (results.size() > 1)
        return success();

    if (auto metadata{ op->getAttrOfType<DictionaryAttr>("metadata") }) {
        auto strAttr{ metadata.getAs<StringAttr>("analysisState") };
        auto strValue{ strAttr.getValue() };
        auto initStateValue{ AlgebraicProperty::Unknown };
        if (strValue.data() != nullptr)
            initStateValue = AlgebraicStructureAnalysisLatticeValue::stringRefAsValue(strValue);
        propagateIfChanged(results[0], results[0]->join(initStateValue));
    } else if (auto matmulOp{ dyn_cast<linalg::MatmulOp>(op) }) {
        const AlgebraicStructureAnalysisLattice* lhsState{ operands[0] };
        const AlgebraicStructureAnalysisLattice* rhsState{ operands[1] };
        auto newValue{ AlgebraicStructureAnalysisLatticeValue::binaryMatmul(lhsState->getValue().getState(), rhsState->getValue().getState()) };
        propagateIfChanged(results[0], results[0]->join(newValue));
    } else if (auto addOp{ dyn_cast<linalg::AddOp>(op) }) {
        const AlgebraicStructureAnalysisLattice* lhsState{ operands[0] };
        const AlgebraicStructureAnalysisLattice* rhsState{ operands[1] };
        auto newValue{ AlgebraicStructureAnalysisLatticeValue::binaryAdd(lhsState->getValue().getState(), rhsState->getValue().getState()) };
        propagateIfChanged(results[0], results[0]->join(newValue));
    } else if (auto elementwiseOp{ dyn_cast<linalg::ElementwiseOp>(op) }) {
        if (operands.size() > 1)
            return success();
        const AlgebraicStructureAnalysisLattice* state{ operands[0] };
        propagateIfChanged(results[0], results[0]->join(state->getValue()));
    } else {
        propagateIfChanged(results[0], results[0]->join(AlgebraicProperty::Unknown));
    }
    
    return success();
}

void AlgebraicStructureAnalysis::setToEntryState(AlgebraicStructureAnalysisLattice *lattice) { 
    auto value{ lattice->getAnchor() };
    llvm::errs() << "=======settoentrystate!!==========\n";
    value.print(llvm::errs());
    llvm::errs() << "\n";
}

}
