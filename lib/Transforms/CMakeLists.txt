add_mlir_library(AlgebraicStructurePasses
    AlgebraicStructureRewrite.cpp
    AlgebraicStructureDebug.cpp

    DEPENDS
    MLIRAlgebraicStructurePasses
    
    LINK_LIBS PUBLIC
    AlgebraicStructureAnalysis
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Linalg)
add_public_tablegen_target(MLIRAlgebraicStructurePasses)
