#include "Analysis/BandedStructureAnalysis.h"

#include <algorithm>

#include "Analysis/BandedProperty.h"
#include "Dialect/DIA/DIAOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::bpa {

LogicalResult BandedStructureAnalysis::run(Block* block) {
    // read block argument attr dictionaries
    if (block->isEntryBlock()) {
        auto* parentOp{ block->getParentOp() };
        if (auto funcOp{ dyn_cast<func::FuncOp>(parentOp) })
            for (auto& arg : block->getArguments())
                if (auto dict{ funcOp.getArgAttrDict(arg.getArgNumber()) }) {
                    auto prop{ readPropertyFromDictAttr(dict) };
                    originalPropertyMap[dyn_cast<Value>(arg)] = prop;
                    propertyMap[dyn_cast<Value>(arg)] = prop;
                }
    }

    for (auto& op : block->getOperations())
        if (failed(visitOperation(&op))) return failure();

    // backward
    return runBackward();
}

LogicalResult BandedStructureAnalysis::runBackward() {
    while (!bwList.empty()) {
        auto* op{ bwList.back() };
        bwList.pop_back();

        SmallVector<Value> inputs;
        if (auto linalgOp{ dyn_cast<linalg::LinalgOp>(op) }) {
            inputs = linalgOp.getDpsInputs();
        } else if (auto diaOp{ dyn_cast<dia::ElementwiseOp>(op) }) {
            inputs = diaOp.getInputs();
        } else {
            continue;  // emptyop or constant op: don't process
        }

        for (auto& v : inputs) {
            auto* definingOp{ v.getDefiningOp() };
            bool validOp{ isa<linalg::LinalgOp>(definingOp) ||
                          isa<dia::ElementwiseOp>(definingOp) ||
                          isa<arith::ConstantOp>(definingOp) || isa<tensor::EmptyOp>(definingOp) ||
                          isa<dia::MatmulOp>(definingOp) || isa<dia::TransposeOp>(definingOp) ||
                          isa<dia::BatchMatmulOp>(definingOp) || isa<dia::SoftmaxOp>(definingOp) };
            if (!validOp || !propagateBackward(v)) continue;
            if (!(isa<linalg::MatmulOp>(definingOp) || isa<dia::MatmulOp>(definingOp) ||
                  isa<linalg::BatchMatmulOp>(definingOp) || isa<dia::BatchMatmulOp>(definingOp)))
                bwList.push_back(definingOp);
        }
    }

    return success();
}

bool BandedStructureAnalysis::propagateBackward(Value value) {
    auto& mat{ propertyMap[value] };
    BandedProperty property{ 0, 0 };

    for (auto* op : value.getUsers()) {
        if (op->getResults().size() != 1) {
            property = mat.Property;
            break;
        }

        auto resultValue{ op->getResult(0) };
        auto resMat{ propertyMap[resultValue] };
        if ((isa<linalg::TransposeOp>(op) || isa<dia::TransposeOp>(op)) &&
            resMat.Property.LowerBandwidth != mat.Property.LowerBandwidth) {
            property = meet(property, BandedProperty{ resMat.Property.LowerBandwidth,
                                                      resMat.Property.UpperBandwidth });
        } else {
            property = meet(property, resMat.Property);
        }
    }

    // return false when no update was made
    if (mat.Property == property) return false;
    mat.Property = join(mat.Property, property);

    return true;
}

BandedSubMatrix BandedStructureAnalysis::readPropertyFromDictAttr(DictionaryAttr dictAttr) {
    BandedSubMatrix generalProp;
    BandedSubMatrix res;

    auto metadataAttr{ dictAttr.get("metadata") };
    if (!metadataAttr) return generalProp;

    auto innerDictAttr{ dyn_cast<DictionaryAttr>(metadataAttr) };
    if (!innerDictAttr) return generalProp;

    auto upperBwAttr{ innerDictAttr.get("upperBw") };
    auto lowerBwAttr{ innerDictAttr.get("lowerBw") };

    if (!upperBwAttr || !lowerBwAttr) return generalProp;

    auto propertyDimsAttr{ innerDictAttr.get("propertyDims") };
    if (!propertyDimsAttr) return generalProp;

    auto propertyDimsArrayAttr{ dyn_cast<ArrayAttr>(propertyDimsAttr) };
    if (!propertyDimsArrayAttr || propertyDimsArrayAttr.size() != 2) {
        return generalProp;
    }

    std::size_t upperBw{ static_cast<std::size_t>(cast<IntegerAttr>(upperBwAttr).getInt()) };
    std::size_t lowerBw{ static_cast<std::size_t>(cast<IntegerAttr>(lowerBwAttr).getInt()) };

    res.Property = BandedProperty(upperBw, lowerBw);
    res.Dims[0] = cast<IntegerAttr>(propertyDimsArrayAttr[0]).getInt();
    res.Dims[1] = cast<IntegerAttr>(propertyDimsArrayAttr[1]).getInt();

    auto diaAttr = innerDictAttr.get("dia");
    if (diaAttr) {
        auto diaBoolAttr = dyn_cast<BoolAttr>(diaAttr);
        res.IsDia = diaBoolAttr.getValue();
    }
    return res;
}

LogicalResult BandedStructureAnalysis::visitOperation(Operation* op) {
    if (op->getNumResults() != 1) return success();
    auto dialect{ op->getDialect() };
    if (!dialect || (dialect->getNamespace() != "linalg" && dialect->getNamespace() != "arith" &&
                     dialect->getNamespace() != "tensor" && dialect->getNamespace() != "dia") &&
                        dialect->getNamespace() != "func") {
        return success();
    }

    auto dict{ op->getAttrDictionary() };
    if (!dict) dict = DictionaryAttr();

    auto result{ op->getResult(0) };
    auto prop{ readPropertyFromDictAttr(dict) };

    originalPropertyMap[result] = prop;
    propertyMap[result] = prop;

    if (auto matmulOp{ dyn_cast<linalg::MatmulOp>(op) }) {
        return visitMatmul(&matmulOp);
    } else if (auto batchMatmulOp{ dyn_cast<linalg::BatchMatmulOp>(op) }) {
        return visitBatchMatmul(&batchMatmulOp);
    } else if (auto addOp{ dyn_cast<linalg::AddOp>(op) }) {
        return visitAdd(&addOp);
    } else if (auto elementwiseOp{ dyn_cast<linalg::ElementwiseOp>(op) }) {
        if (elementwiseOp.getKind() == linalg::ElementwiseKind::mul) bwList.push_back(op);
        return visitElementwise(&elementwiseOp);
    } else if (auto mulOp{ dyn_cast<linalg::MulOp>(op) }) {
        bwList.push_back(op);
        return visitMul(&mulOp);
    } else if (auto transposeOp{ dyn_cast<linalg::TransposeOp>(op) }) {
        return visitTranspose(&transposeOp);
    } else if (auto genericOp{ dyn_cast<linalg::GenericOp>(op) }) {
        return visitGeneric(&genericOp);
    } else if (auto diaMatmulOp{ dyn_cast<dia::MatmulOp>(op) }) {
        return visitDIAMatmul(&diaMatmulOp);
    } else if (auto diaFromDenseOp{ dyn_cast<dia::FromDenseOp>(op) }) {
        return visitDIAFromDense(&diaFromDenseOp);
    } else if (auto diaBatchMatmulOp{ dyn_cast<dia::BatchMatmulOp>(op) }) {
        return visitDIABatchMatmul(&diaBatchMatmulOp);
    } else if (auto diaElementwiseOp{ dyn_cast<dia::ElementwiseOp>(op) }) {
        if (diaElementwiseOp.getKind() == dia::ElementwiseKind::mul) bwList.push_back(op);
        return visitDIAElementwise(&diaElementwiseOp);
    } else if (auto diaTransposeOp{ dyn_cast<dia::TransposeOp>(op) }) {
        return visitDIATranspose(&diaTransposeOp);
    } else if (auto diaSoftmaxOp{ dyn_cast<dia::SoftmaxOp>(op) }) {
        return visitDIASoftmax(&diaSoftmaxOp);
    }

    return success();
}

LogicalResult BandedStructureAnalysis::visitDIAFromDense(dia::FromDenseOp* op) {
    auto input = op->getInput();
    auto result = op->getResult();

    if (!propertyMap.contains(input)) return failure();
    auto inputProperty = propertyMap[input];
    inputProperty.IsDia = true;
    propertyMap[result] = inputProperty;
    return success();
}

LogicalResult BandedStructureAnalysis::visitDIAMatmul(dia::MatmulOp* op) {
    auto lhs = op->getLhs();
    auto rhs = op->getRhs();
    auto result = op->getResult();

    auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
    auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());

    if (!propertyMap.contains(lhs) || !propertyMap.contains(rhs)) return success();

    auto lhsShape{ lhsType.getShape() };
    auto rhsShape{ rhsType.getShape() };

    const auto& lhsMat = propertyMap[lhs];
    const auto& rhsMat = propertyMap[rhs];

    BandedProperty newProperty{ binaryMatmul(lhsMat.Property, rhsMat.Property) };

    // WARNING: only works for squared matrices. True shape is (shape[1] x shape[1])
    newProperty.UpperBandwidth = std::min<uint64_t>(newProperty.UpperBandwidth, rhsShape[1] - 1);
    newProperty.LowerBandwidth = std::min<uint64_t>(newProperty.LowerBandwidth, lhsShape[1] - 1);

    BandedSubMatrix& resMat = propertyMap[result];
    resMat.Property = join(resMat.Property, newProperty);
    resMat.Dims[0] = lhsMat.Dims[0];
    resMat.Dims[1] = rhsMat.Dims[1];
    resMat.IsDia = detectDIA ? false : true;

    return success();
}

LogicalResult BandedStructureAnalysis::visitDIABatchMatmul(dia::BatchMatmulOp* op) {
    auto operands{ op->getOperands() };
    auto result{ op->getResult() };
    auto lhs{ op->getLhs() };
    auto rhs{ op->getRhs() };
    auto lhsType{ dyn_cast<TensorType>(lhs.getType()) };
    auto rhsType{ dyn_cast<TensorType>(rhs.getType()) };

    auto lhsShape{ lhsType.getShape() };
    auto rhsShape{ rhsType.getShape() };

    if (!propertyMap.contains(lhs) || !propertyMap.contains(rhs)) return success();

    const auto& lhsMat{ propertyMap[lhs] };
    const auto& rhsMat{ propertyMap[rhs] };

    std::array<uint64_t, 2> expectedDims{ 1, 2 };
    if (lhsMat.Dims != expectedDims || rhsMat.Dims != expectedDims) return success();

    BandedProperty newProperty{ binaryMatmul(lhsMat.Property, rhsMat.Property) };

    newProperty.UpperBandwidth = std::min<uint64_t>(newProperty.UpperBandwidth, rhsShape[2] - 1);
    newProperty.LowerBandwidth = std::min<uint64_t>(newProperty.LowerBandwidth, lhsShape[2] - 1);

    propertyMap[result] = BandedSubMatrix{ newProperty, { 1, 2 } };
    propertyMap[result].IsDia = detectDIA ? false : true;

    return success();
}

LogicalResult BandedStructureAnalysis::visitDIAElementwise(dia::ElementwiseOp* op) {
    auto operands{ op->getOperands() };
    auto result{ op->getResult() };

    auto lhs{ operands[0] };
    if (!propertyMap.contains(lhs)) return success();

    const auto& lhsMat{ propertyMap[lhs] };

    auto& resMat{ propertyMap[result] };

    if (operands.size() == 2) {  // output matrix counts as an operand
        auto inputProp{ propertyMap[operands[0]] };
        auto newProperty{ unaryElementwise(inputProp.Property) };
        resMat = { join(propertyMap[result].Property, newProperty), lhsMat.Dims };
        return success();
    }

    auto rhs{ operands[1] };
    if (!propertyMap.contains(rhs)) return success();

    const auto& rhsMat{ propertyMap[rhs] };

    if (rhsMat.Dims[0] != lhsMat.Dims[0] || rhsMat.Dims[1] != lhsMat.Dims[1]) return success();

    if (operands.size() != 3) return failure();
    if (op->getKind() == dia::ElementwiseKind::mul) {
        auto newProperty{ binaryElementwiseProduct(lhsMat.Property, rhsMat.Property) };
        resMat = { join(propertyMap[result].Property, newProperty), propertyMap[operands[0]].Dims };
    } else {
        auto newProperty{ binaryElementwiseGeneral(lhsMat.Property, rhsMat.Property) };
        resMat = { join(propertyMap[result].Property, newProperty), propertyMap[operands[0]].Dims };
    }

    resMat.IsDia = detectDIA ? false : true;
    return success();
}

LogicalResult BandedStructureAnalysis::visitDIATranspose(dia::TransposeOp* op) {
    auto input = op->getInput();
    auto result = op->getResult();

    if (!propertyMap.contains(input)) return failure();

    const BandedSubMatrix inputBand = propertyMap[input];
    const BandedProperty newProperty(inputBand.Property.LowerBandwidth,
                                     inputBand.Property.UpperBandwidth);
    // doesn't currently support higher dims
    propertyMap[result] = BandedSubMatrix{ newProperty, { 0, 1 }, true };

    return success();
}

LogicalResult BandedStructureAnalysis::visitDIASoftmax(dia::SoftmaxOp* op) {
    auto input = op->getInput();
    auto result = op->getResult();

    if (!propertyMap.contains(input)) return failure();

    const BandedSubMatrix inputBand = propertyMap[input];
    propertyMap[result] = inputBand;

    return success();
}

LogicalResult BandedStructureAnalysis::visitMatmul(linalg::MatmulOp* op) {
    auto operands{ op->getOperands() };
    auto result{ op->getResult(0) };
    auto lhs{ operands[0] };
    auto rhs{ operands[1] };
    auto lhsType{ dyn_cast<RankedTensorType>(lhs.getType()) };
    auto rhsType{ dyn_cast<RankedTensorType>(rhs.getType()) };

    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) return success();

    auto lhsShape{ lhsType.getShape() };
    auto rhsShape{ rhsType.getShape() };

    if (!propertyMap.contains(lhs) || !propertyMap.contains(rhs)) return success();

    const auto& lhsMat{ propertyMap[lhs] };
    const auto& rhsMat{ propertyMap[rhs] };

    BandedProperty newProperty{ binaryMatmul(lhsMat.Property, rhsMat.Property) };
    newProperty.UpperBandwidth = std::min<uint64_t>(newProperty.UpperBandwidth, rhsShape[1] - 1);
    newProperty.LowerBandwidth = std::min<uint64_t>(newProperty.LowerBandwidth, lhsShape[0] - 1);

    BandedSubMatrix& resMat{ propertyMap[result] };
    resMat.Property = join(resMat.Property, newProperty);

    auto resultType{ dyn_cast<RankedTensorType>(result.getType()) };
    if (resultType && resultType.hasStaticShape()) {
        resMat.Dims[0] = lhsMat.Dims[0];
        resMat.Dims[1] = rhsMat.Dims[1];
    }

    return success();
}

LogicalResult BandedStructureAnalysis::visitBatchMatmul(linalg::BatchMatmulOp* op) {
    auto operands{ op->getOperands() };
    auto result{ op->getResult(0) };
    auto lhs{ operands[0] };
    auto rhs{ operands[1] };
    auto lhsType{ dyn_cast<TensorType>(lhs.getType()) };
    auto rhsType{ dyn_cast<TensorType>(rhs.getType()) };

    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) return success();

    auto lhsShape{ lhsType.getShape() };
    auto rhsShape{ rhsType.getShape() };

    if (!propertyMap.contains(lhs) || !propertyMap.contains(rhs)) return success();

    const auto& lhsMat{ propertyMap[lhs] };
    const auto& rhsMat{ propertyMap[rhs] };

    std::array<uint64_t, 2> expectedDims{ 1, 2 };
    if (lhsMat.Dims != expectedDims || rhsMat.Dims != expectedDims) return success();

    BandedProperty newProperty{ binaryMatmul(lhsMat.Property, rhsMat.Property) };
    newProperty.UpperBandwidth = std::min<uint64_t>(newProperty.UpperBandwidth, rhsShape[2] - 1);
    newProperty.LowerBandwidth = std::min<uint64_t>(newProperty.LowerBandwidth, lhsShape[1] - 1);

    propertyMap[result] = BandedSubMatrix{ newProperty, { 1, 2 } };

    return success();
}

LogicalResult BandedStructureAnalysis::visitAdd(linalg::AddOp* op) {
    for (auto& map : op->getIndexingMapsArray())
        if (!map.isIdentity()) return success();

    auto operands{ op->getOperands() };
    auto result{ op->getResult(0) };
    auto lhs{ operands[0] };
    auto rhs{ operands[1] };

    if (!propertyMap.contains(lhs) || !propertyMap.contains(rhs)) return success();

    const auto& lhsMat{ propertyMap[lhs] };
    const auto& rhsMat{ propertyMap[rhs] };

    if (lhsMat.Dims != rhsMat.Dims) return success();

    BandedProperty resProp{ binaryElementwiseGeneral(lhsMat.Property, rhsMat.Property) };

    propertyMap[result] = { join(propertyMap[result].Property, resProp), lhsMat.Dims };

    return success();
}

LogicalResult BandedStructureAnalysis::visitMul(linalg::MulOp* op) {
    for (auto& map : op->getIndexingMapsArray())
        if (!map.isIdentity()) return success();

    auto result{ op->getResult(0) };
    auto operands{ op->getOperands() };
    auto lhs{ operands[0] };
    auto rhs{ operands[1] };

    if (!propertyMap.contains(lhs) || !propertyMap.contains(rhs)) return success();

    const auto& lhsMat{ propertyMap[lhs] };
    const auto& rhsMat{ propertyMap[rhs] };

    if (lhsMat.Dims != rhsMat.Dims) return success();
    auto newProperty{ binaryElementwiseProduct(lhsMat.Property, rhsMat.Property) };
    propertyMap[result] = { join(propertyMap[result].Property, newProperty),
                            propertyMap[operands[0]].Dims };
    return success();
}

LogicalResult BandedStructureAnalysis::visitTranspose(linalg::TransposeOp* op) {
    auto input = op->getDpsInputOperand(0)->get();
    auto resultRange{ op->getResult() };
    auto result{ resultRange[0] };

    if (!propertyMap.contains(input)) return success();
    auto inputType{ dyn_cast<RankedTensorType>(input.getType()) };
    if (!inputType || inputType.getShape().size() < 2) return success();

    auto permutation{ op->getPermutation() };
    const auto& mat{ propertyMap[input] };

    auto dim0Iter{ std::find(permutation.begin(), permutation.end(), mat.Dims[0]) };
    auto dim0{ static_cast<uint64_t>(std::distance(permutation.begin(), dim0Iter)) };
    auto dim1Iter{ std::find(permutation.begin(), permutation.end(), mat.Dims[1]) };
    auto dim1{ static_cast<uint64_t>(std::distance(permutation.begin(), dim1Iter)) };

    assert(dim0Iter != permutation.end() && dim1Iter != permutation.end());

    auto& resMat{ propertyMap[result] };
    if (dim0 < dim1) {
        auto newProperty{ mat.Property };
        propertyMap[result] = { join(resMat.Property, newProperty), { dim0, dim1 } };
        return success();
    }

    auto newProperty{ transpose(mat.Property) };
    propertyMap[result] = { join(resMat.Property, newProperty), { dim1, dim0 } };

    return success();
}

LogicalResult BandedStructureAnalysis::visitElementwise(linalg::ElementwiseOp* op) {
    for (auto& map : op->getIndexingMapsArray())
        if (!map.isIdentity()) return success();

    auto operands{ op->getOperands() };
    auto lhs{ operands[0] };
    if (!propertyMap.contains(lhs)) return success();

    const auto& lhsMat{ propertyMap[lhs] };

    auto result{ op->getResult(0) };
    auto& resMat{ propertyMap[result] };

    if (operands.size() == 2) {  // output matrix counts as an operand
        auto inputProp{ propertyMap[operands[0]] };
        auto newProperty{ unaryElementwise(inputProp.Property) };
        resMat = { join(propertyMap[result].Property, newProperty), lhsMat.Dims };
        return success();
    }

    auto rhs{ operands[1] };
    if (!propertyMap.contains(rhs)) return success();

    const auto& rhsMat{ propertyMap[rhs] };

    if (rhsMat.Dims[0] != lhsMat.Dims[0] || rhsMat.Dims[1] != lhsMat.Dims[1]) return success();

    if (op->getKind() == linalg::ElementwiseKind::mul) {
        auto newProperty{ binaryElementwiseProduct(lhsMat.Property, rhsMat.Property) };
        resMat = { join(propertyMap[result].Property, newProperty), propertyMap[operands[0]].Dims };
    } else {
        auto newProperty{ binaryElementwiseGeneral(lhsMat.Property, rhsMat.Property) };
        resMat = { join(propertyMap[result].Property, newProperty), propertyMap[operands[0]].Dims };
    }

    return success();
}

LogicalResult BandedStructureAnalysis::visitGeneric(linalg::GenericOp* op) {
    if (op->getNumDpsInputs() != 2 || op->getNumDpsInits() != 1) return success();

    auto inputs{ op->getInputs() };
    auto lhs{ inputs[0] };
    auto rhs{ inputs[1] };

    auto lhsType{ cast<RankedTensorType>(lhs.getType()) };
    auto rhsType{ cast<RankedTensorType>(rhs.getType()) };

    if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape()) return success();

    auto lhsShape{ lhsType.getShape() };
    auto rhsShape{ rhsType.getShape() };

    auto result{ op->getResult(0) };

    if (!propertyMap.contains(lhs) || !propertyMap.contains(rhs)) return success();

    const auto& lhsMat{ propertyMap[lhs] };
    const auto& rhsMat{ propertyMap[rhs] };

    auto indexingMaps{ op->getIndexingMapsArray() };
    if (indexingMaps.size() != 3) return success();

    auto lhsMap{ indexingMaps[0] };
    auto rhsMap{ indexingMaps[1] };
    auto resultMap{ indexingMaps[2] };

    if (!lhsMap.isProjectedPermutation() || !rhsMap.isProjectedPermutation() ||
        !resultMap.isProjectedPermutation())
        return success();

    auto mExpr{ lhsMap.getResult(lhsMat.Dims[0]) };
    auto kExprLhs{ lhsMap.getResult(lhsMat.Dims[1]) };
    auto kExprRhs{ rhsMap.getResult(rhsMat.Dims[0]) };
    auto nExpr{ rhsMap.getResult(rhsMat.Dims[1]) };

    if (kExprLhs != kExprRhs) return success();

    std::optional<int> mResultIdx;
    std::optional<int> nResultIdx;

    for (unsigned i = 0; i < resultMap.getNumResults(); ++i) {
        auto resExpr{ resultMap.getResult(i) };
        if (resExpr == mExpr) mResultIdx = static_cast<int>(i);
        if (resExpr == nExpr) nResultIdx = static_cast<int>(i);
    }

    if (!mResultIdx.has_value() || !nResultIdx.has_value()) return success();

    auto iterTypes{ op->getIteratorTypesArray() };
    auto kDimPos{ cast<AffineDimExpr>(kExprLhs).getPosition() };
    auto mDimPos{ cast<AffineDimExpr>(mExpr).getPosition() };
    auto nDimPos{ cast<AffineDimExpr>(nExpr).getPosition() };

    if (iterTypes[kDimPos] != utils::IteratorType::reduction) return success();
    if (iterTypes[mDimPos] != utils::IteratorType::parallel) return success();
    if (iterTypes[nDimPos] != utils::IteratorType::parallel) return success();

    auto mSize{ lhsType.getShape()[lhsMat.Dims[0]] };
    auto kSize{ lhsType.getShape()[lhsMat.Dims[1]] };
    auto nSize{ rhsType.getShape()[rhsMat.Dims[1]] };

    Block* body{ op->getBody() };
    if (body->getNumArguments() != 3) return success();

    auto terminator{ dyn_cast<linalg::YieldOp>(body->getTerminator()) };
    if (!terminator || terminator->getNumOperands() != 1) return success();

    Operation* addOp{ terminator->getOperand(0).getDefiningOp() };
    if (!addOp || !(isa<arith::AddIOp>(addOp) || isa<arith::AddFOp>(addOp))) return success();

    auto accArg{ body->getArgument(2) };
    Value mulResult;

    if (addOp->getOperand(0) == accArg)
        mulResult = addOp->getOperand(1);
    else if (addOp->getOperand(1) == accArg)
        mulResult = addOp->getOperand(0);
    else
        return success();

    Operation* mulOp{ mulResult.getDefiningOp() };
    if (!mulOp || !(isa<arith::MulIOp>(mulOp) || isa<arith::MulFOp>(mulOp))) return success();

    auto arg0{ body->getArgument(0) };
    auto arg1{ body->getArgument(1) };
    bool isMulArgsValid{ (mulOp->getOperand(0) == arg0 && mulOp->getOperand(1) == arg1) ||
                         (mulOp->getOperand(0) == arg1 && mulOp->getOperand(1) == arg0) };

    if (!isMulArgsValid) return success();

    BandedProperty newProperty{ binaryMatmul(lhsMat.Property, rhsMat.Property) };
    newProperty.LowerBandwidth = std::min<uint64_t>(newProperty.LowerBandwidth, mSize - 1);
    newProperty.UpperBandwidth = std::min<uint64_t>(newProperty.UpperBandwidth, nSize - 1);

    propertyMap[result] = BandedSubMatrix{
        newProperty, { static_cast<uint64_t>(*mResultIdx), static_cast<uint64_t>(*nResultIdx) }
    };

    return success();
}

}  // namespace mlir::bpa
