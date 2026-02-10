#ifndef LIB_TRANSFORMS_ALGEBRAICSTRUCTUREREWRITE_H
#define LIB_TRANSFORMS_ALGEBRAICSTRUCTUREREWRITE_H
#include "mlir/Pass/Pass.h"

namespace mlir::asa {
#define GEN_PASS_DECL_ALGEBRAICSTRUCTUREREWRITE
#include "lib/Transforms/Passes.h.inc"
}  // namespace mlir::asa
#endif
