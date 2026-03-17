set(LLVM_TARGET_DEFINITIONS DIADialect.td)
mlir_tablegen(DIADialect.h.inc -gen-dialect-decls)
mlir_tablegen(DIADialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRDIADialectIncGen)

set(LLVM_TARGET_DEFINITIONS DIAOps.td)
mlir_tablegen(DIAOps.h.inc -gen-op-decls)
mlir_tablegen(DIAOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRDIAOpsIncGen)

set(LLVM_TARGET_DEFINITIONS DIAEnums.td)
mlir_tablegen(DIAEnums.h.inc -gen-enum-decls)
mlir_tablegen(DIAEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRDIAEnumsIncGen)

set(LLVM_TARGET_DEFINITIONS DIAAttrs.td)
mlir_tablegen(DIAAttrs.h.inc -gen-attrdef-decls)
mlir_tablegen(DIAAttrs.cpp.inc -gen-attrdef-defs)
add_public_tablegen_target(MLIRDIAAttrsIncGen)

