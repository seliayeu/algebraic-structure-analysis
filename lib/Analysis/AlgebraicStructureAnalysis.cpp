#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "lib/Analysis/AlgebraicStructureAnalysis.h"

namespace mlir::asa {

LogicalResult AlgebraicStructureAnalysis::run(Block* block) {
    if (block->isEntryBlock()) {
        auto* parentOp{ block->getParentOp() };
        if (auto funcOp{ dyn_cast<func::FuncOp>(parentOp) })
            for (auto& arg : block->getArguments())
                if (auto dict{ funcOp.getArgAttrDict(arg.getArgNumber()) })
                    propertyMap[dyn_cast<Value>(arg)] = readPropertyFromDictAttr(dict);
    }

    for (auto& op : block->getOperations()) {
        if (failed(visitOperation(&op)))
            return failure();
    }

    return success();
}

LogicalResult AlgebraicStructureAnalysis::visitOperation(Operation *op) {
    if (op->getNumResults() != 1)
        return success();
    auto dialect{ op->getDialect() };
    if (!dialect || (dialect->getNamespace() != "linalg"
        && dialect->getNamespace() != "arith"
        && dialect->getNamespace() != "tensor")) {
        return success();
    }

    auto dict{ op->getAttrDictionary() };
    if (!dict) 
        dict = DictionaryAttr();
    
    auto result{ op->getResult(0) };
    propertyMap[result] = readPropertyFromDictAttr(dict);

    if (auto matmulOp{ dyn_cast<linalg::MatmulOp>(op) }) {
        auto operands{ matmulOp->getOperands() };
        auto lhs{ operands[0] };
        auto rhs{ operands[1] };
        auto lhsType{ dyn_cast<TensorType>(lhs.getType()) };
        auto rhsType{ dyn_cast<TensorType>(rhs.getType()) };
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

        auto newValue{ binaryMatmul(propertyMap[lhs], propertyMap[rhs]) };
        propertyMap[result] = 
            join(propertyMap[result], newValue);
    } else if (auto addOp{ dyn_cast<linalg::AddOp>(op) }) {
        auto operands{ addOp.getOperands() };
        auto lhs{ operands[0] };
        auto rhs{ operands[1] };
        auto lhsType{ dyn_cast<TensorType>(lhs.getType()) };
        auto rhsType{ dyn_cast<TensorType>(rhs.getType()) };
        auto lhsShape{ lhsType.getShape() };
        auto rhsShape{ rhsType.getShape() };
        if (!(lhsShape[0] == lhsShape[1]) || !(rhsShape[0] == rhsShape[1])) 
            return success();
        auto newValue{ binaryAdd(propertyMap[lhs], propertyMap[rhs]) };
        propertyMap[result] = 
            join(propertyMap[result], newValue);
    } else if (auto elementwiseOp{ dyn_cast<linalg::ElementwiseOp>(op)}) {
        for (auto& map : elementwiseOp.getIndexingMapsArray())
            if (!map.isIdentity()) 
                return success();
        auto operands{ elementwiseOp.getOperands() };
        auto newValue{ AlgebraicProperty::General };
        if (operands.size() == 1)
            newValue = unaryElementwise(propertyMap[operands[0]]);
        else if (operands.size() == 2) {
            newValue = binaryElementwiseGeneral(propertyMap[operands[0]],
                propertyMap[operands[1]]);
            propertyMap[result] = 
                join(propertyMap[result], newValue);
        }
    } else if (auto mulOp{ dyn_cast<linalg::MulOp>(op)}) {
        for (auto& map : mulOp.getIndexingMapsArray())
            if (!map.isIdentity()) 
                return success();
        auto operands{ mulOp.getOperands() };
        auto newValue{ binaryElementwiseProduct(propertyMap[operands[0]],
            propertyMap[operands[1]])};
        propertyMap[result] = 
            join(propertyMap[result], newValue);
    } else if (auto transposeOp{ dyn_cast<linalg::TransposeOp>(op)}) {
        auto operands{ transposeOp.getOperands() };
        auto permutation{ transposeOp.getPermutation() };
        if (permutation.size() != 2 || !(permutation[0] == 1 && permutation[1] == 0) )
            return success();
        auto newValue{ transpose(propertyMap[operands[0]]) };
        propertyMap[result] = 
            join(propertyMap[result], newValue);
    } 

    return success();
}

}
