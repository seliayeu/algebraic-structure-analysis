#ifndef LIB_ANALYSIS_BANDEDSTRUCTUREANALYSIS_H
#define LIB_ANALYSIS_BANDEDSTRUCTUREANALYSIS_H

#include <array>
#include <limits>
#include <vector>

#include "Analysis/BandedProperty.h"
#include "Dialect/DIA/DIAOps.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::bpa {

struct BandedSubMatrix {
    BandedProperty Property{ std::numeric_limits<int64_t>::max(),
                             std::numeric_limits<int64_t>::max() };

    // TODO: chekc if Dims are being used
    std::array<uint64_t, 2> Dims{ 0, 1 };

    bool IsDia{ false };

    bool operator==(const BandedSubMatrix& other) const {
        return (Property == other.Property) && (Dims == other.Dims) && (IsDia == other.IsDia);
    }

    bool operator!=(const BandedSubMatrix& other) const {
        return !operator==(other);
    }
};

class BandedStructureAnalysis {
    llvm::DenseMap<Value, BandedSubMatrix> propertyMap;
    std::vector<Operation*> bwList;  // list of ops to perform bw prop on

    bool detectDIA{ false };

   public:
    LogicalResult run(Block* block);

    BandedSubMatrix getProperty(Value value) {
        return propertyMap[value];
    }

    bool hasProperty(Value value) {
        return propertyMap.contains(value);
    }

    static BandedSubMatrix readPropertyFromDictAttr(DictionaryAttr dictAttr);

   private:
    LogicalResult visitOperation(Operation* op);

    // DIA ops
    LogicalResult visitMatmul(dia::MatmulOp* op);
    // Linalg ops
    LogicalResult visitMatmul(linalg::MatmulOp* op);
    LogicalResult visitBatchMatmul(linalg::BatchMatmulOp* op);
    LogicalResult visitAdd(linalg::AddOp* addOp);
    LogicalResult visitElementwise(linalg::ElementwiseOp* elementwiseOp);
    LogicalResult visitMul(linalg::MulOp* mulOp);
    LogicalResult visitTranspose(linalg::TransposeOp* transposeOp);
    LogicalResult visitGeneric(linalg::GenericOp* transposeOp);

    // Func Ops
    LogicalResult visitFuncCall(func::CallOp* funcCallOp);

    LogicalResult runBackward();
    bool propagateBackward(Value value);
};

}  // namespace mlir::bpa

#endif
