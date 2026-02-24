#ifndef LIB_TRANSFORMS_BANDEDREWRITE_H
#define LIB_TRANSFORMS_BANDEDREWRITE_H
#include "mlir/Pass/Pass.h"

namespace mlir::bpa {
#define GEN_PASS_DECL_BANDEDREWRITE
#include "lib/Transform/Passes.h.inc"
}  // namespace mlir::bpa
#endif
