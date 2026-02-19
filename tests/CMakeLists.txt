add_lit_testsuite(check-asp
  "Running the asp_tester tests"
  ${CMAKE_CURRENT_SOURCE_DIR}/
  DEPENDS alg-opt
)

# set_property(TARGET check-asp
#   PROPERTY ENVIRONMENT
#   "PATH=${LLVM_TOOLS_BINARY_DIR}:$ENV{PATH}"
# )
#

set_property(TARGET check-asp
  APPEND PROPERTY LIT_ARGS
  "-v"  # optional, verbose
  "--param=PATH=${LLVM_TOOLS_BINARY_DIR}:$ENV{PATH}"
)
