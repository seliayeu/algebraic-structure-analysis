set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Banded)
add_public_tablegen_target(MLIRBandedStructurePasses)

add_mlir_library(BandedStructurePasses
    BandedRewrite.cpp
    BandedPropagation.cpp
    LinalgRewrite.cpp
    DenseSoftmaxRewrite.cpp
    Operators/DIAMatmulRewrite.cpp
    Operators/DIATransposeRewrite.cpp
    Operators/DIAElementwiseRewrite.cpp
    Operators/DIABatchMatmulRewrite.cpp
    Operators/DIASoftmaxRewrite.cpp
    DEPENDS
    MLIRDIADialectIncGen
    LINK_LIBS PUBLIC
    MLIRDIADialect
    BandedStructureAnalysis
    MLIRPass
    MLIRTransforms
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRTensorDialect
    MLIRSCFDialect
)

target_include_directories(BandedStructurePasses PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
