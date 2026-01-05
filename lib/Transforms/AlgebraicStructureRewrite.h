#ifndef LIB_TRANSFORMS_ALGEBRAICSTRUCTUREREWRITE_H
#define LIB_TRANSFORMS_ALGEBRAICSTRUCTUREREWRITE_H
#include "mlir/Pass/Pass.h"

namespace mlir::asa {
#define GEN_PASS_DECL_ALGEBRAICSTRUCTUREREWRITE
#define GEN_PASS_REGISTRATION
#include "lib/Transforms/AlgebraicStructureRewrite.h.inc"
}
#endif
