#ifndef LIB_TRANSFORMS_LINALGREWRITE_H
#define LIB_TRANSFORMS_LINALGREWRITE_H
#include "mlir/Pass/Pass.h"

namespace mlir::bpa {
#define GEN_PASS_DECL_LINALGREWRITE
#include "lib/Transform/Passes.h.inc"
}  // namespace mlir::bpa
#endif
