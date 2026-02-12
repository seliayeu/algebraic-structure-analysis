#ifndef LIB_ANALYSIS_ALGEBRAICSTRUCTUREANALYSIS_H
#define LIB_ANALYSIS_ALGEBRAICSTRUCTUREANALYSIS_H
#include <array>

#include "lib/Analysis/AlgebraicProperty.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::asa {

struct SubMatrixProperty {
    AlgebraicProperty property;
    std::array<long long, 2> dimensions{ -1, -1 };
    bool operator==(const SubMatrixProperty& other) const {
        return (property == other.property) && (dimensions == other.dimensions);
    }
    bool operator!=(const SubMatrixProperty& other) const {
        return !operator==(other);
    }
};

class AlgebraicStructureAnalysis {
    llvm::DenseMap<mlir::Value, SubMatrixProperty> propertyMap;

   public:
    LogicalResult run(Block* block);
    SubMatrixProperty getProperty(mlir::Value value) {
        return propertyMap[value];
    }

    bool hasProperty(mlir::Value value) {
        return propertyMap.contains(value);
    }

   private:
    LogicalResult visitOperation(Operation* op);
    LogicalResult visitMatmul(linalg::MatmulOp*);
    LogicalResult visitBatchMatmul(linalg::BatchMatmulOp*);
    LogicalResult visitAdd(linalg::AddOp* addOp);
    LogicalResult visitElementwise(linalg::ElementwiseOp* elementwiseOp);
    LogicalResult visitMul(linalg::MulOp* mulOp);
    LogicalResult visitTranspose(linalg::TransposeOp* transposeOp);
    LogicalResult visitGeneric(linalg::GenericOp* transposeOp);
    SubMatrixProperty readPropertyFromDictAttr(DictionaryAttr dictAttr);
};
}  // namespace mlir::asa

#endif
