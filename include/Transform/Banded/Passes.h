#ifndef LIB_TRANSFORMS_PASSES_H
#define LIB_TRANSFORMS_PASSES_H
#include "Transform/Banded/BandedStructureDebug.h"
#include "mlir/Pass/Pass.h"

namespace mlir::bpa {
#define GEN_PASS_REGISTRATION_BANDEDSTRUCTUREDEBUG
#include "Passes.h.inc"
}  // namespace mlir::bpa
#endif
