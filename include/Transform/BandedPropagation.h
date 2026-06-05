#ifndef LIB_TRANSFORMS_BANDEDPROPAGATION_H
#define LIB_TRANSFORMS_BANDEDPROPAGATION_H
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::bpa {
#define GEN_PASS_DECL_BANDEDANALYSIS
#include "lib/Transform/Passes.h.inc"
}  // namespace mlir::bpa

struct BandedAnalysisResult {
    BandedAnalysisResult(mlir::func::FuncOp op) {
    }

   public:
    bool detectDIA = false;
    bool disableBw = false;
};

#endif
