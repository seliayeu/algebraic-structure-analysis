#ifndef LIB_TRANSFORMS_PASSES_H
#define LIB_TRANSFORMS_PASSES_H
#include "lib/Transforms/AlgebraicStructureDebug.h"
#include "lib/Transforms/AlgebraicStructureRewrite.h"
#include "mlir/Pass/Pass.h"

namespace mlir::asa {
#define GEN_PASS_REGISTRATION
#include "lib/Transforms/Passes.h.inc"
}  // namespace mlir::asa
#endif
