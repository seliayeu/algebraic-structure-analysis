#ifndef LIB_TRANSFORMS_LINALGMATMULREWRITE_H
#define LIB_TRANSFORMS_LINALGMATMULREWRITE_H
#include "mlir/Pass/Pass.h"

namespace mlir::bpa {
#define GEN_PASS_DECL_LINALGMATMULREWRITE
#include "lib/Transform/Passes.h.inc"
}  // namespace mlir::bpa
#endif
