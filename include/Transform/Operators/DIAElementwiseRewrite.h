
#pragma once
#include "mlir/IR/PatternMatch.h"

namespace mlir::bpa {
void addDIAElementwisePatterns(RewritePatternSet& patterns);
}  // namespace mlir::bpa
