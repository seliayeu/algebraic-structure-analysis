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

LogicalResult AlgebraicStructureAnalysis::initialize(Operation* top) {
    for (Region &region : top->getRegions()) {
        if (region.empty())
            continue;
        for (Block &block : region.getBlocks()) {
            if (block.empty())
                continue;
            for (Operation &op : block.getOperations()) {
                if (failed(initializeOperation(&op)))
                    return failure();
            }
        }
    }

    return success();
}

LogicalResult AlgebraicStructureAnalysis::initializeOperation(Operation* op) {
    if (auto metadata{ op->getAttrOfType<DictionaryAttr>("metadata") }) {
        auto strAttr{ metadata.getAs<StringAttr>("analysisState") };
        auto strValue{ strAttr.getValue() };
        auto initStateValue{ AlgebraicStructureAnalysisState::ASAValue::Unknown };
        if (strValue.data() != nullptr)
            initStateValue = AlgebraicStructureAnalysisState::stringAsValue(strValue);
        (void)getOrCreate<AlgebraicStructureAnalysisState>(getProgramPointAfter(op))->setValue(initStateValue);

    } else {
        (void)getOrCreate<AlgebraicStructureAnalysisState>(getProgramPointAfter(op))->setValue(AlgebraicStructureAnalysisState::ASAValue::Unknown);
    }

    if (op->getNumRegions() && failed(initialize(op)))
        return failure();

    if (failed(visit(getProgramPointAfter(op))))
        return failure();

    return success();
}

LogicalResult AlgebraicStructureAnalysis::visit(ProgramPoint* point) {
    auto op{ point->getPrevOp() };
    
    llvm::errs() << "=======Visiting==========";
    op->dump();
    llvm::errs() << "\n";

    if (auto matmulOp{ dyn_cast<linalg::MatmulOp>(op) }) {
        // if (matmulOp.getIndexingMaps()) // don't support non-standard indexing
        //     return success();
        auto matmulInputs{ matmulOp.getDpsInputs() };
        auto lhs{ matmulInputs[0] };
        auto lhsDefiningOp{ lhs.getDefiningOp() };
        auto rhs{ matmulInputs[1] };
        auto rhsDefiningOp{ rhs.getDefiningOp() };

        if (!lhsDefiningOp || !rhsDefiningOp)
            return success();

        auto lhsState{ getOrCreate<AlgebraicStructureAnalysisState>(getProgramPointAfter(lhsDefiningOp)) };
        auto rhsState{ getOrCreate<AlgebraicStructureAnalysisState>(getProgramPointAfter(rhsDefiningOp)) };
        auto newValue{ AlgebraicStructureAnalysisState::binMatmul(lhsState->getValue(), rhsState->getValue()) };
        auto state{ getOrCreate<AlgebraicStructureAnalysisState>(getProgramPointAfter(op)) };
        propagateIfChanged(state, state->setValue(newValue));
    } else if (auto addOp{ dyn_cast<linalg::AddOp>(op) }) {
        auto addInputs{ addOp.getInputs() };
        auto lhs{ addInputs[0] };
        auto lhsDefiningOp{ lhs.getDefiningOp() };
        auto rhs{ addInputs[1] };
        auto rhsDefiningOp{ rhs.getDefiningOp() };

        if (!lhsDefiningOp || !rhsDefiningOp)
            return success();

        auto lhsState{ getOrCreate<AlgebraicStructureAnalysisState>(getProgramPointAfter(lhsDefiningOp)) };
        auto rhsState{ getOrCreate<AlgebraicStructureAnalysisState>(getProgramPointAfter(rhsDefiningOp)) };
        auto newValue{ AlgebraicStructureAnalysisState::binAdd(lhsState->getValue(), rhsState->getValue()) };
        auto state{ getOrCreate<AlgebraicStructureAnalysisState>(getProgramPointAfter(op)) };
        propagateIfChanged(state, state->setValue(newValue));
    }
    
    return success();
}


AlgebraicStructureAnalysis::AlgebraicStructureAnalysis(DataFlowSolver &solver) : DataFlowAnalysis(solver) {
    registerAnchorKind<ProgramPoint>();
}

}
