#ifndef LIB_TRANSFORMS_PASSES_H
#define LIB_TRANSFORMS_PASSES_H
#include "lib/Transforms/BandedStructureDebug.h"
#include "mlir/Pass/Pass.h"

namespace mlir::bpa {
#define GEN_PASS_REGISTRATION_BANDEDSTRUCTUREDEBUG
#include "lib/Transforms/Passes.h.inc"
}  // namespace mlir::bpa
#endif
