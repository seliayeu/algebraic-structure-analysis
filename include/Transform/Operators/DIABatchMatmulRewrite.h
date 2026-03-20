
#pragma once
#include "mlir/IR/PatternMatch.h"

namespace mlir::bpa {
void addDIABatchMatmulPatterns(RewritePatternSet& patterns);
}  // namespace mlir::bpa
