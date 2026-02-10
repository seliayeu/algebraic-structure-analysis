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


LogicalResult AlgebraicStructureAnalysis::visitMatmul(linalg::MatmulOp* op) {
    auto operands{ op->getOperands() };
    auto result{ op->getResult(0) };
    auto lhs{ operands[0] };
    auto rhs{ operands[1] };
    auto lhsType{ dyn_cast<TensorType>(lhs.getType()) };
    auto rhsType{ dyn_cast<TensorType>(rhs.getType()) };
    auto lhsShape{ lhsType.getShape() };
    auto rhsShape{ rhsType.getShape() };

    if (!(lhsShape[0] == lhsShape[1]) || !(rhsShape[0] == rhsShape[1])) 
        return success();

    auto indexingMapArr{ op->getIndexingMapsArray() };
    if (indexingMapArr.size() != 3)
        return success();

    auto* ctx{ op->getContext() };
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

LogicalResult AlgebraicStructureAnalysis::visitBatchMatmul(linalg::BatchMatmulOp* op) {
    auto operands{ op->getOperands() };
    auto result{ op->getResult(0) };
    auto lhs{ operands[0] };
    auto rhs{ operands[1] };
    auto lhsType{ dyn_cast<TensorType>(lhs.getType()) };
    auto rhsType{ dyn_cast<TensorType>(rhs.getType()) };

    auto lhsShape{ lhsType.getShape() };
    auto rhsShape{ rhsType.getShape() };

    if (!(lhsShape[1] == lhsShape[2]) || !(rhsShape[1] == rhsShape[2])) 
        return success();

    auto indexingMapArr{ op->getIndexingMapsArray() };
    if (indexingMapArr.size() != 3)
        return success();

    auto* ctx{ op->getContext() };
    AffineExpr b, m, n, k;
    bindDims(ctx, b, m, n, k);
    
    auto expectedMapLhs{ AffineMap::get(4, 0, {b, m, k}, ctx) };
    auto expectedMapRhs{ AffineMap::get(4, 0, {b, k, n}, ctx) };
    auto expectedMapRes{ AffineMap::get(4, 0, {b, m, n}, ctx) };

    if (indexingMapArr[0] != expectedMapLhs || 
        indexingMapArr[1] != expectedMapRhs || 
        indexingMapArr[2] != expectedMapRes) {
        return success();
    }

    if (!(propertyMap[lhs].dimensions[0] == 1 && propertyMap[lhs].dimensions[1] == 2)
            || !(propertyMap[rhs].dimensions[0] == 1 && propertyMap[rhs].dimensions[1] == 2))
        return success();

    auto newProperty{ binaryMatmul(propertyMap[lhs].property, propertyMap[rhs].property) };
    newProperty = join(propertyMap[result].property, newProperty);
    
    propertyMap[result] = SubMatrixProperty{ newProperty, { 1, 2 } };

    return success();
}

LogicalResult AlgebraicStructureAnalysis::visitAdd(linalg::AddOp* op) {
    auto operands{ op->getOperands() };
    auto result{ op->getResult(0) };
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

LogicalResult AlgebraicStructureAnalysis::visitMul(linalg::MulOp* op) {
    for (auto& map : op->getIndexingMapsArray())
        if (!map.isIdentity()) 
            return success();
    auto operands{ op->getOperands() };
    auto result{ op->getResult(0) };
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

LogicalResult AlgebraicStructureAnalysis::visitTranspose(linalg::TransposeOp* op) {
    auto operands{ op->getOperands() };
    auto resultRange{ op->getResult() };
    auto result{ resultRange[0] };
    auto permutation{ op->getPermutation() };
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

LogicalResult AlgebraicStructureAnalysis::visitElementwise(linalg::ElementwiseOp* op) {
    for (auto& map : op->getIndexingMapsArray())
        if (!map.isIdentity()) 
            return success();

    auto operands{ op->getOperands() };
    auto result{ op->getResult(0) };

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

LogicalResult AlgebraicStructureAnalysis::visitGeneric(linalg::GenericOp* op) {
    auto inputs{ op->getInputs() };
    auto outputs{ op->getOutputs() };

    auto lhs{ inputs[0] };
    auto rhs{ inputs[1] };
    auto result{ outputs[0] };

    auto& lhsProp{ propertyMap[lhs] };
    auto& rhsProp{ propertyMap[rhs] };

    // verify indexing maps
    auto indexingMaps{ op->getIndexingMapsArray() };
    auto lhsMap{ indexingMaps[0] };
    auto rhsMap{ indexingMaps[1] };
    auto resultMap{ indexingMaps[2] };

    if (!lhsMap.isProjectedPermutation() ||  !rhsMap.isProjectedPermutation() 
            || !resultMap.isProjectedPermutation())
        return success();

    auto mExpr{ lhsMap.getResult(lhsProp.dimensions[0]) };
    auto kExpr{ lhsMap.getResult(lhsProp.dimensions[1]) };
    auto nExpr{ rhsMap.getResult(rhsProp.dimensions[1]) };

    if (kExpr != rhsMap.getResult(rhsProp.dimensions[0]))
        return success();

    std::optional<int> mResultIdx;
    std::optional<int> nResultIdx;

    for (unsigned i = 0; i < resultMap.getNumResults(); ++i) {
        auto resExpr{ resultMap.getResult(i) };
        if (resExpr == mExpr) mResultIdx = static_cast<int>(i);
        if (resExpr == nExpr) nResultIdx = static_cast<int>(i);
    }

    if (!mResultIdx.has_value() || !nResultIdx.has_value())
        return success();

    // verify iterators
    auto iterTypes{ op->getIteratorTypesArray() };
    auto kDimPos{ cast<AffineDimExpr>(kExpr).getPosition() };

    if (iterTypes[kDimPos] != utils::IteratorType::reduction)
        return success();

    auto mDimPos{ cast<AffineDimExpr>(mExpr).getPosition() };
    auto nDimPos{ cast<AffineDimExpr>(nExpr).getPosition() };

    if (iterTypes[mDimPos] != utils::IteratorType::parallel 
            ||  iterTypes[nDimPos] != utils::IteratorType::parallel)
        return success();

    auto body{ op->getBody() };
    if (body->getOperations().size() != 3 || body->getNumArguments() != 3)
        return success();

    auto accumulator{ body->getArgument(2) };

    auto yieldOp{ cast<linalg::YieldOp>(body->getTerminator()) };
    auto yieldOperands{ yieldOp->getOperands() };
    if (yieldOperands.size() != 1)
        return success();
    auto addOp{ yieldOperands[0].getDefiningOp() };
    if (!addOp || !isa<arith::AddIOp, arith::AddFOp>(addOp))
        return success();
    Operation* mulOp;
    if (addOp->getOperand(0) == accumulator) {
        mulOp = addOp->getOperand(1).getDefiningOp();
    } else if (addOp->getOperand(1) == accumulator) {
        mulOp = addOp->getOperand(0).getDefiningOp();
    } else {
        return success();
    }

    if (!mulOp || !isa<arith::MulIOp, arith::MulFOp>(mulOp))
        return success();

    if (!((mulOp->getOperand(0) == body->getArgument(1) && mulOp->getOperand(1) == body->getArgument(0)) 
            || (mulOp->getOperand(1) == body->getArgument(0) && mulOp->getOperand(0) == body->getArgument(1))))
        return success();

    auto newProperty{ binaryMatmul(lhsProp.property, rhsProp.property) };
    newProperty = join(propertyMap[result].property, newProperty);

    propertyMap[result] = SubMatrixProperty{ 
        newProperty, 
        { mResultIdx.value(), nResultIdx.value() } 
    };

    return success();
}

}
