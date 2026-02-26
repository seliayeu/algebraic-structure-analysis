add_mlir_library(BandedUtils
    TransformUtils.cpp

    LINK_LIBS PUBLIC
    BandedStructureAnalysis
)

