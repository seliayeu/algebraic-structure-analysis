#include "Dialect/DIA/DIADialect.h"

#include "Dialect/DIA/DIAOps.h"
#include "llvm/ADT/TypeSwitch.h"

// clang-format off
#include "Dialect/DIA/DIADialect.cpp.inc"
#include "Dialect/DIA/DIAEnums.cpp.inc"
#define GET_ATTRDEF_CLASSES
#include "Dialect/DIA/DIAAttrs.cpp.inc"
// clang-format on

namespace mlir::bpa::dia {

void DIADialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "Dialect/DIA/DIAOps.cpp.inc"
        >();
    addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/DIA/DIAAttrs.cpp.inc"
        >();
}
}  // namespace mlir::bpa::dia
