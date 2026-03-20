
#pragma once
#include "mlir/IR/PatternMatch.h"

namespace mlir::bpa {
void addDIATransposePatterns(RewritePatternSet& patterns);
}  // namespace mlir::bpa
