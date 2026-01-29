#ifndef LIB_TRANSFORMS_ALGEBRAICSTRUCTUREDEBUG_H
#define LIB_TRANSFORMS_ALGEBRAICSTRUCTUREDEBUG_H
#include "mlir/Pass/Pass.h"

namespace mlir::asa {
#define GEN_PASS_DECL_ALGEBRAICSTRUCTUREDEBUG
#include "lib/Transforms/Passes.h.inc"
}
#endif
