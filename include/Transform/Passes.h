#ifndef LIB_TRANSFORMS_PASSES_H
#define LIB_TRANSFORMS_PASSES_H
#include "Transform/BandedPropagation.h"

namespace mlir::bpa {
#define GEN_PASS_REGISTRATION_BANDEDANALYSIS
#include "Passes.h.inc"
}  // namespace mlir::bpa
#endif
