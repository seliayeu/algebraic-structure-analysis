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

SubMatrixProperty AlgebraicStructureAnalysis::readPropertyFromDictAttr(DictionaryAttr dictAttr) {
    auto metadataAttr{ dictAttr.get("metadata") };
    if (!metadataAttr)
        return { AlgebraicProperty::General };
    auto innerDictAttr{ dyn_cast<DictionaryAttr>(metadataAttr) };
    if (!innerDictAttr)
        return { AlgebraicProperty::General };
    auto analysisPropertyAttr{ innerDictAttr.get("analysisState") };
    if (!analysisPropertyAttr)
        return { AlgebraicProperty::General };
    auto analysisProperty{ dyn_cast<StringAttr>(analysisPropertyAttr).getValue() };

    auto propertyDimsAttr{ innerDictAttr.get("propertyDims") };
    if (!propertyDimsAttr)
        return { AlgebraicProperty::General };
    auto propertyDimsArrayAttr{ dyn_cast<ArrayAttr>(propertyDimsAttr) };
    if (!propertyDimsArrayAttr || propertyDimsArrayAttr.size() != 2) {
        return { AlgebraicProperty::General };
    }
    SubMatrixProperty res{ stringToValue(std::string(analysisProperty)) }; 
    res.dimensions[0] = cast<IntegerAttr>(propertyDimsArrayAttr[0]).getInt();
    res.dimensions[1] = cast<IntegerAttr>(propertyDimsArrayAttr[1]).getInt();

    return res;
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
        return visitMatmul(&matmulOp);
    } else if (auto addOp{ dyn_cast<linalg::AddOp>(op) }) {
        return visitAdd(&addOp);
    } else if (auto elementwiseOp{ dyn_cast<linalg::ElementwiseOp>(op)}) {
        return visitElementwise(&elementwiseOp);
    } else if (auto mulOp{ dyn_cast<linalg::MulOp>(op)}) {
        return visitMul(&mulOp);
    } else if (auto transposeOp{ dyn_cast<linalg::TransposeOp>(op)}) {
        return visitTranspose(&transposeOp);
    } 

    return success();
}


LogicalResult AlgebraicStructureAnalysis::visitMatmul(linalg::MatmulOp* matmulOp) {
    auto operands{ matmulOp->getOperands() };
    auto result{ matmulOp->getResult(0) };
    auto lhs{ operands[0] };
    auto rhs{ operands[1] };
    auto lhsType{ dyn_cast<TensorType>(lhs.getType()) };
    auto rhsType{ dyn_cast<TensorType>(rhs.getType()) };
    auto lhsShape{ lhsType.getShape() };
    auto rhsShape{ rhsType.getShape() };

    if (!(lhsShape[0] == lhsShape[1]) || !(rhsShape[0] == rhsShape[1])) 
        return success();

    auto indexingMapArr{ matmulOp->getIndexingMapsArray() };
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

    auto newProperty{ binaryMatmul(propertyMap[lhs].property, propertyMap[rhs].property) };
    newProperty = join(propertyMap[result].property, newProperty);
    propertyMap[result] = SubMatrixProperty{ newProperty, { 0, 1 } };

    return success();
}

LogicalResult AlgebraicStructureAnalysis::visitAdd(linalg::AddOp* addOp) {
    auto operands{ addOp->getOperands() };
    auto result{ addOp->getResult(0) };
    auto lhs{ operands[0] };
    auto rhs{ operands[1] };
    auto lhsType{ dyn_cast<TensorType>(lhs.getType()) };
    auto rhsType{ dyn_cast<TensorType>(rhs.getType()) };
    auto lhsShape{ lhsType.getShape() };
    auto rhsShape{ rhsType.getShape() };
    if (!(lhsShape[0] == lhsShape[1]) || !(rhsShape[0] == rhsShape[1])) 
        return success();
    
    if (propertyMap[lhs].dimensions != propertyMap[rhs].dimensions)
        return success();
    AlgebraicProperty newProperty;
    newProperty = binaryElementwiseGeneral(propertyMap[lhs].property, propertyMap[rhs].property);
    propertyMap[result] = {
        join(propertyMap[result].property, newProperty),
        propertyMap[operands[0]].dimensions
    };
    return success();
}

LogicalResult AlgebraicStructureAnalysis::visitMul(linalg::MulOp* mulOp) {
    for (auto& map : mulOp->getIndexingMapsArray())
        if (!map.isIdentity()) 
            return success();
    auto operands{ mulOp->getOperands() };
    auto result{ mulOp->getResult(0) };
    if (propertyMap[operands[0]].dimensions != propertyMap[operands[1]].dimensions)
        return success();
    auto newProperty{ binaryElementwiseProduct(propertyMap[operands[0]].property,
        propertyMap[operands[1]].property)};
    propertyMap[result] = {
        join(propertyMap[result].property, newProperty),
        propertyMap[operands[0]].dimensions
    };
    return success();
}

LogicalResult AlgebraicStructureAnalysis::visitTranspose(linalg::TransposeOp* transposeOp) {
    auto operands{ transposeOp->getOperands() };
    auto resultRange{ transposeOp->getResult() };
    auto result{ resultRange[0] };
    auto permutation{ transposeOp->getPermutation() };
    if (permutation.size() != 2)
        return success();
    if (permutation[0] != propertyMap[operands[0]].dimensions[1]
            && permutation[1] != propertyMap[operands[0]].dimensions[0])
        return success();

    auto newProperty{ transpose(propertyMap[operands[0]].property) };
    propertyMap[result] = {
        join(propertyMap[result].property, newProperty),
        { propertyMap[operands[0]].dimensions[1], propertyMap[operands[0]].dimensions[0] }
    };
    return success();
}

LogicalResult AlgebraicStructureAnalysis::visitElementwise(linalg::ElementwiseOp* elementwiseOp) {
    for (auto& map : elementwiseOp->getIndexingMapsArray())
        if (!map.isIdentity()) 
            return success();

    auto operands{ elementwiseOp->getOperands() };
    auto result{ elementwiseOp->getResult(0) };

    if (operands.size() == 1) {
        auto newProperty{ unaryElementwise(propertyMap[operands[0]].property) };
        propertyMap[result].property = join(propertyMap[result].property, newProperty);
    } else if (operands.size() == 2) {
        if (propertyMap[operands[0]].dimensions != propertyMap[operands[1]].dimensions)
            return success();
        auto newProperty{ binaryElementwiseGeneral(propertyMap[operands[0]].property,
            propertyMap[operands[1]].property) };
        newProperty = join(propertyMap[result].property, newProperty);
        propertyMap[result] = {
            join(propertyMap[result].property, newProperty),
            propertyMap[operands[0]].dimensions
        };
    }
    return success();
}

}
