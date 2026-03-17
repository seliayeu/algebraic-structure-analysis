#ifndef DIA_ENUMS_TD
#define DIA_ENUMS_TD

include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/EnumAttr.td"

def UnaryFn : I32EnumAttr<"UnaryFn", "", [
  I32EnumAttrCase<"square", 0>,
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::bpa::dia";
}

def BinaryFn : I32EnumAttr<"BinaryFn", "", [
  I32EnumAttrCase<"add", 1>,
  I32EnumAttrCase<"sub", 2>,
  I32EnumAttrCase<"mul", 3>,
]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::bpa::dia";
}

def ElementwiseKind :I32EnumAttr<"ElementwiseKind",
                        "", !listconcat(UnaryFn.enumerants, BinaryFn.enumerants)
                      > {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::bpa::dia";
}

#endif
