add_mlir_dialect_library(MLIRDIADialect
    DIADialect.cpp
    DIAOps.cpp
    DEPENDS
    MLIRDIADialectIncGen
    MLIRDIAOpsIncGen
    MLIRDIAAttrsIncGen
    MLIRDIAEnumsIncGen
    LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
)

target_include_directories(MLIRDIADialect PUBLIC
    $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
)

