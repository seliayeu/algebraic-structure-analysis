#ifndef LIB_ANALYSIS_BANDEDSTRUCTUREANALYSIS_H
#define LIB_ANALYSIS_BANDEDSTRUCTUREANALYSIS_H

#include <array>
#include <cstddef>
#include <limits>

#include "lib/Analysis/BandedProperty.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::bpa {

struct BandedSubMatrix {
    BandedProperty Property{ std::numeric_limits<long>::max(), std::numeric_limits<long>::max() };
    std::array<long long, 2> Dims{ 0, 1 };

    bool operator==(const BandedSubMatrix& other) const {
        return (Property == other.Property) && (Dims == other.Dims);
    }
    bool operator!=(const BandedSubMatrix& other) const {
        return !operator==(other);
    }
};

class BandedStructureAnalysis {
    llvm::DenseMap<mlir::Value, BandedSubMatrix> propertyMap;

   public:
    LogicalResult run(Block* block);

    BandedSubMatrix getProperty(mlir::Value value) {
        return propertyMap[value];
    }

    bool hasProperty(mlir::Value value) {
        return propertyMap.contains(value);
    }

   private:
    LogicalResult visitOperation(Operation* op);
    LogicalResult visitMatmul(linalg::MatmulOp* op);
    LogicalResult visitBatchMatmul(linalg::BatchMatmulOp* op);
    LogicalResult visitAdd(linalg::AddOp* addOp);
    LogicalResult visitElementwise(linalg::ElementwiseOp* elementwiseOp);
    LogicalResult visitMul(linalg::MulOp* mulOp);
    LogicalResult visitTranspose(linalg::TransposeOp* transposeOp);
    LogicalResult visitGeneric(linalg::GenericOp* transposeOp);

    BandedSubMatrix readPropertyFromDictAttr(DictionaryAttr dictAttr);
};

}  // namespace mlir::bpa

#endif
