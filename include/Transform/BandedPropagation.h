#ifndef LIB_TRANSFORMS_BANDEDSTRUCTUREDEBUG_H
#define LIB_TRANSFORMS_BANDEDSTRUCTUREDEBUG_H
#include "mlir/Pass/Pass.h"

namespace mlir::bpa {
#define GEN_PASS_DECL_BANDEDANALYSIS
#include "lib/Transform/Passes.h.inc"
}  // namespace mlir::bpa
#endif
