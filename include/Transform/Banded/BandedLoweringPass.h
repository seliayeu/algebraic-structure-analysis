#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::bpa {

class BandedLoweringPass : public PassWrapper<BandedLoweringPass, OperationPass<func::FuncOp>> {
   public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BandedLoweringPass)

    StringRef getArgument() const final {
        return "banded-lowering";
    }

    StringRef getDescription() const override {
        return "Lower linalg.ops to efficient implementation based on band "
               "information";
    }

    void runOnOperation() override;
};

void registerBandedLoweringPass();
}  // namespace mlir::bpa
