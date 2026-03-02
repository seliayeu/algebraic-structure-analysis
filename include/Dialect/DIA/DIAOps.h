#ifndef DIA_OPS_H
#define DIA_OPS_H

#pragma once
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// clang-format off
#include "Dialect/DIA/DIADialect.h"
// clang-format on

#define GET_OP_CLASSES
#include "Dialect/DIA/DIAOps.h.inc"

#endif
