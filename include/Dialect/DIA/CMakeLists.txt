set(LLVM_TARGET_DEFINITIONS DIADialect.td)
mlir_tablegen(DIADialect.h.inc -gen-dialect-decls)
mlir_tablegen(DIADialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRDIADialectIncGen)

set(LLVM_TARGET_DEFINITIONS DIAOps.td)
mlir_tablegen(DIAOps.h.inc -gen-op-decls)
mlir_tablegen(DIAOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRDIAOpsIncGen)
