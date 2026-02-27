#include "Dialect/DIA/DIADialect.h"

#include "Dialect/DIA/DIADialect.cpp.inc"
#include "Dialect/DIA/DIAOps.h"

namespace mlir::bpa::dia {

void DIADialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "Dialect/DIA/DIAOps.cpp.inc"
        >();
}
}  // namespace mlir::bpa::dia
