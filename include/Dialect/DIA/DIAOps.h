#ifndef DIA_OPS_H
#define DIA_OPS_H
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// clang-format off
#include "Dialect/DIA/DIADialect.h"
// clang-format on

#include "Dialect/DIA/DIAEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "Dialect/DIA/DIAAttrs.h.inc"
#define GET_OP_CLASSES
#include "Dialect/DIA/DIAOps.h.inc"

#endif
