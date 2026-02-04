#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "lib/Analysis/AlegbraicStructureAnalysis.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "llvm/Support/DebugLog.h"
#include <iostream>

namespace mlir::asa {

LogicalResult AlgebraicStructureAnalysis::visitOperation(Operation *op,
                                       ArrayRef<const AlgebraicStructureAnalysisLattice*> operands,
                                       ArrayRef<AlgebraicStructureAnalysisLattice*> results) {
    if (results.size() > 1)
        return success();

    if (auto metadata{ op->getAttrOfType<DictionaryAttr>("metadata") }) {
        auto strAttr{ metadata.getAs<StringAttr>("analysisState") };
        auto strValue{ strAttr.getValue() };
        auto initStateValue{ AlgebraicProperty::General};
        if (strValue.data() != nullptr)
            initStateValue = AlgebraicStructureAnalysisLatticeValue::stringRefAsValue(strValue);
        propagateIfChanged(results[0], results[0]->join(initStateValue));
    } else if (auto matmulOp{ dyn_cast<linalg::MatmulOp>(op) }) {
        auto matmulOperands{ matmulOp->getOperands() };
        auto lhsType{ dyn_cast<TensorType>(matmulOperands[0].getType()) };
        auto rhsType{ dyn_cast<TensorType>(matmulOperands[1].getType()) };
        auto lhsShape{ lhsType.getShape() };
        auto rhsShape{ rhsType.getShape() };
        if (!(lhsShape[0] == lhsShape[1]) || !(rhsShape[0] == rhsShape[1])) 
            return success();

        auto indexingMapArr{ matmulOp.getIndexingMapsArray() };
        if (indexingMapArr.size() != 3)
            return success();

        auto* ctx{ matmulOp->getContext() };
        AffineExpr m, n, k;
        bindDims(ctx, m, n, k);
        auto expectedMapLhs{ AffineMap::get(3, 0, {m, k}, ctx) };
        auto expectedMapRhs{ AffineMap::get(3, 0, {k, n}, ctx) };
        auto expectedMapRes{ AffineMap::get(3, 0, {m, n}, ctx) };

        if (indexingMapArr[0] != expectedMapLhs || 
            indexingMapArr[1] != expectedMapRhs || 
            indexingMapArr[2] != expectedMapRes) {
            return success();
        }

        const AlgebraicStructureAnalysisLattice* lhsState{ operands[0] };
        const AlgebraicStructureAnalysisLattice* rhsState{ operands[1] };
        auto newValue{ AlgebraicStructureAnalysisLatticeValue::binaryMatmul(lhsState->getValue().getState(), rhsState->getValue().getState()) };
        propagateIfChanged(results[0], results[0]->join(newValue));
    } else if (auto addOp{ dyn_cast<linalg::AddOp>(op) }) {
        auto addOperands{ addOp.getOperands() };
        auto lhsType{ dyn_cast<TensorType>(addOperands[0].getType()) };
        auto rhsType{ dyn_cast<TensorType>(addOperands[1].getType()) };
        auto lhsShape{ lhsType.getShape() };
        auto rhsShape{ rhsType.getShape() };
        if (!(lhsShape[0] == lhsShape[1]) || !(rhsShape[0] == rhsShape[1])) 
            return success();
        const AlgebraicStructureAnalysisLattice* lhsState{ operands[0] };
        const AlgebraicStructureAnalysisLattice* rhsState{ operands[1] };
        auto newValue{ AlgebraicStructureAnalysisLatticeValue::binaryAdd(lhsState->getValue().getState(), rhsState->getValue().getState()) };
        propagateIfChanged(results[0], results[0]->join(newValue));
    } else if (auto elementwiseOp{ dyn_cast<linalg::ElementwiseOp>(op)}) {
        for (auto& map : elementwiseOp.getIndexingMapsArray())
            if (!map.isIdentity()) 
                return success();
        if (operands.size() == 1) {
            auto newValue{ AlgebraicStructureAnalysisLatticeValue::unaryElementwise(operands[0]->getValue().getState()) };
            propagateIfChanged(results[0], results[0]->join(newValue));
        } else if (operands.size() == 2) {
            auto newValue{ elementwiseOp.getKind() == linalg::ElementwiseKind::mul ?
                AlgebraicStructureAnalysisLatticeValue::binaryElementwiseProduct(operands[0]->getValue().getState(), operands[1]->getValue().getState()) :
                AlgebraicStructureAnalysisLatticeValue::binaryElementwiseGeneral(operands[0]->getValue().getState(), operands[1]->getValue().getState()) };
            propagateIfChanged(results[0], results[0]->join(newValue));
        }
    } else if (auto mulOp{ dyn_cast<linalg::MulOp>(op)}) {
        for (auto& map : mulOp.getIndexingMapsArray())
            if (!map.isIdentity()) 
                return success();
        auto newValue{ AlgebraicStructureAnalysisLatticeValue::binaryElementwiseProduct(operands[0]->getValue().getState(), operands[1]->getValue().getState()) };
        propagateIfChanged(results[0], results[0]->join(newValue));
    } else if (auto transposeOp{ dyn_cast<linalg::TransposeOp>(op)}) {
        auto permutation{ transposeOp.getPermutation() };
        if (permutation.size() != 2 || !(permutation[0] == 1 && permutation[1] == 0) )
            return success();
        auto newValue{ AlgebraicStructureAnalysisLatticeValue::transpose(operands[0]->getValue().getState()) };
        propagateIfChanged(results[0], results[0]->join(newValue));
    } else if (isa<RegionBranchOpInterface>(op) || isa<BranchOpInterface>(op)) {
        return success();
    } else if (!results.empty()) {
        propagateIfChanged(results[0], results[0]->join(AlgebraicProperty::General));
    }
    
    return success();
}

void AlgebraicStructureAnalysis::setToEntryState(AlgebraicStructureAnalysisLattice *lattice) { 
    auto value{ lattice->getAnchor() };
    auto blockArg{ dyn_cast<BlockArgument>(value) };
    if (!blockArg) return;
    auto* block{ blockArg.getOwner() };
    if (!block->isEntryBlock()) return;

    auto op{ block->getParentOp() };
    auto funcOp{ dyn_cast<func::FuncOp>(op) };
    if (!funcOp) return;
    auto attrDict{ funcOp.getArgAttrDict(blockArg.getArgNumber()) };
    if (!attrDict) {
        propagateIfChanged(lattice, lattice->join(AlgebraicProperty::General));
        return;
    }

    auto strAttr{ attrDict.getAs<StringAttr>("analysisState") };
    auto strValue{ strAttr.getValue() };
    auto initStateValue{ AlgebraicProperty::General};
    if (strValue.data() != nullptr)
        initStateValue = AlgebraicStructureAnalysisLatticeValue::stringRefAsValue(strValue);
    propagateIfChanged(lattice, lattice->join(initStateValue));
}
}

