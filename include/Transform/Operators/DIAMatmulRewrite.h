#pragma once
#include "mlir/IR/PatternMatch.h"

namespace mlir::bpa {
void addDIAMatmulPatterns(RewritePatternSet& patterns, bool detectDia);
}  // namespace mlir::bpa
