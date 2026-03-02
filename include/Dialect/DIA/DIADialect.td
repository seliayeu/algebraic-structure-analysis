#ifndef DIA_DIALECT_TD
#define DIA_DIALECT_TD

include "mlir/IR/DialectBase.td"

def DIA_Dialect : Dialect {
    let name = "dia";
    let summary = "Dialect for DIA banded matrix operations";
    let cppNamespace = "::mlir::bpa::dia";
}

#endif // DIA_DIALECT
