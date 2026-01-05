add_mlir_library(AlgebraicStructureRewrite
    AlgebraicStructureRewrite.cpp

    DEPENDS
    MLIRAlgebraicStructureRewrite
    
    LINK_LIBS PUBLIC
)

set(LLVM_TARGET_DEFINITIONS AlgebraicStructureRewrite.td)
mlir_tablegen(AlgebraicStructureRewrite.h.inc -gen-pass-decls -name Linalg)
add_public_tablegen_target(MLIRAlgebraicStructureRewrite)
