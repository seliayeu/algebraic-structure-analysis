#ifndef LIB_ANALYSIS_BANDEDSTRUCTUREANALYSIS_H
#define LIB_ANALYSIS_BANDEDSTRUCTUREANALYSIS_H

#include <array>
#include <limits>
#include <vector>

#include "Analysis/BandedProperty.h"
#include "Dialect/DIA/DIAOps.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::bpa {

struct BandedSubMatrix {
    BandedProperty Property{ std::numeric_limits<int64_t>::max(),
                             std::numeric_limits<int64_t>::max() };

    std::array<uint64_t, 2> Dims{ 0, 1 };

    bool IsDia{ false };

    bool operator==(const BandedSubMatrix& other) const {
        return (Property == other.Property) && (Dims == other.Dims) && (IsDia == other.IsDia);
    }

    bool operator!=(const BandedSubMatrix& other) const {
        return !operator==(other);
    }

    bool isDiagonal() const {
        return Property.LowerBandwidth == 0 && Property.UpperBandwidth == 0;
    }
};

class BandedStructureAnalysis {
    llvm::DenseMap<Value, BandedSubMatrix> propertyMap;
    std::vector<Operation*> bwList;  // list of ops to perform bw prop on

    bool detectDIA{ false };

   public:
    BandedStructureAnalysis(bool detectDIA) {
        this->detectDIA = detectDIA;
    }

    LogicalResult run(Block* block);

    BandedSubMatrix getProperty(Value value) {
        return propertyMap[value];
    }

    bool hasProperty(Value value) {
        return propertyMap.contains(value);
    }

    static BandedSubMatrix readPropertyFromDictAttr(DictionaryAttr dictAttr);

   private:
    // DIA ops
    LogicalResult visitMatmul(dia::MatmulOp* op);
    LogicalResult visitFromDense(dia::FromDenseOp* op);
    LogicalResult visitDIABatchMatmul(dia::BatchMatmulOp* op);
    LogicalResult visitDIAElementwise(dia::ElementwiseOp* op);
    LogicalResult visitTranspose(dia::TransposeOp* op);

    // Linalg ops
    LogicalResult visitOperation(Operation* op);
    LogicalResult visitMatmul(linalg::MatmulOp* op);
    LogicalResult visitBatchMatmul(linalg::BatchMatmulOp* op);
    LogicalResult visitAdd(linalg::AddOp* addOp);
    LogicalResult visitElementwise(linalg::ElementwiseOp* elementwiseOp);
    LogicalResult visitMul(linalg::MulOp* mulOp);
    LogicalResult visitTranspose(linalg::TransposeOp* transposeOp);
    LogicalResult visitGeneric(linalg::GenericOp* transposeOp);

    LogicalResult runBackward();
    bool propagateBackward(Value value);
};

}  // namespace mlir::bpa

#endif
