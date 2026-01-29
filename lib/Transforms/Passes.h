#ifndef LIB_TRANSFORMS_PASSES_H
#define LIB_TRANSFORMS_PASSES_H
#include "mlir/Pass/Pass.h"
#include "lib/Transforms/AlgebraicStructureRewrite.h"
#include "lib/Transforms/AlgebraicStructureDebug.h"

namespace mlir::asa {
#define GEN_PASS_REGISTRATION
#include "lib/Transforms/Passes.h.inc"
}
#endif
