set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Banded)
add_public_tablegen_target(MLIRBandedStructurePasses)

add_mlir_library(BandedStructurePasses
    BandedPropagation.cpp
    BandedRewrite.cpp

    DEPENDS
    MLIRBandedStructurePasses

    LINK_LIBS PUBLIC
    BandedStructureAnalysis
)

target_include_directories(BandedStructurePasses PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
