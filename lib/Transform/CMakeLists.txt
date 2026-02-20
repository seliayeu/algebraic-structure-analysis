set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Linalg)
add_public_tablegen_target(MLIRBandedStructurePasses)

add_mlir_library(BandedStructurePasses
    BandedPropagation.cpp
    BandedLoweringPass.cpp

    DEPENDS
    MLIRBandedStructurePasses

    LINK_LIBS PUBLIC
    BandedStructureAnalysis
)

target_include_directories(BandedStructurePasses PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
