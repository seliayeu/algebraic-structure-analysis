add_lit_testsuite(check-bpa
  "Running the asp_tester tests"
  ${CMAKE_CURRENT_SOURCE_DIR}/
  DEPENDS alg-opt
)

set_property(TARGET check-bpa
  APPEND PROPERTY LIT_ARGS
  "-v"
  "--param=PATH=${LLVM_TOOLS_BINARY_DIR}:$ENV{PATH}"
)
