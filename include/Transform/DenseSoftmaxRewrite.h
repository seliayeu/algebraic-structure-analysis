#ifndef LIB_TRANSFORMS_DENSESOFTMAXREWRITE_H
#define LIB_TRANSFORMS_DENSESOFTMAXREWRITE_H
#include "mlir/Pass/Pass.h"

namespace mlir::bpa {
#define GEN_PASS_DECL_DENSESOFTMAXREWRITE
#include "lib/Transform/Passes.h.inc"
}  // namespace mlir::bpa
#endif
