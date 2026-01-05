get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set (LIBS
    ${dialect_libs}
    ${conversion_libs}
    MLIROptLib
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    AlgebraicStructureRewrite
)

add_llvm_executable(alg-opt alg-opt.cpp)

llvm_update_compile_flags(alg-opt)
target_link_libraries(alg-opt PRIVATE ${LIBS})

mlir_check_all_link_libraries(alg-opt)
