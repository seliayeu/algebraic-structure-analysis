#ifndef LIB_TRANSFORMS_PASSES_H
#define LIB_TRANSFORMS_PASSES_H
#include "Transform/BandedPropagation.h"
#include "Transform/BandedRewrite.h"

namespace mlir::bpa {
#define GEN_PASS_REGISTRATION
#include "lib/Transform/Passes.h.inc"
}  // namespace mlir::bpa
#endif
