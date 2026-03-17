#ifndef DIA_ATTRS_TD
#define DIA_ATTRS_TD

include "mlir/IR/CommonTypeConstraints.td"
include "DIAEnums.td"
include "DIADialect.td"

def ElementwiseKindAttr : EnumAttr<DIA_Dialect,
                                   ElementwiseKind, "elementwise_kind"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif
