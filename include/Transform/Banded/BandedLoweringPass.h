#pragma once
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::bpa {

class BandLoweringPass : public PassWrapper<BandLoweringPass, OperationPass<func::FuncOp>> {
   public:
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BandLoweringPass)

    StringRef getArgument() const final {
        return "band-lowering";
    }

    StringRef getDescription() const override {
        return "Lower linalg.ops to efficient implementation based on band "
               "information";
    }

    void runOnOperation() override;
};

void registerBandLoweringPass();
}  // namespace mlir::bpa
