add_mlir_library(BandedStructurePasses
    BandedStructureDebug.cpp

    DEPENDS
    MLIRBandedStructurePasses
    
    LINK_LIBS PUBLIC
    BandedStructureAnalysis
)

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Linalg)
add_public_tablegen_target(MLIRBandedStructurePasses)
