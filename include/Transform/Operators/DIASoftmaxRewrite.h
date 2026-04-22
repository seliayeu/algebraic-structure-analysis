
#pragma once
#include "mlir/IR/PatternMatch.h"

namespace mlir::bpa {
void addDIASoftmaxPatterns(RewritePatternSet& patterns);
}
