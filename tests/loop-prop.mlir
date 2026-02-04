// RUN: ../build/tools/alg-opt %s --algebraic-structure-debug | FileCheck %s
module {
  func.func private @printMemrefF32(memref<*xf32>)

  // CHECK-LABEL: func.func @main
  func.func @main() {
    // CHECK: arith.constant
    // CHECK-SAME: analysisState = "Symmetric"
    %sym_const = arith.constant { metadata = { analysisState = "Symmetric" } } 
      dense<[[2.0, 4.0, 5.0], 
             [4.0, 3.0, 9.0], 
             [5.0, 9.0, 7.0]]> : tensor<3x3xf32>

    // CHECK: arith.constant
    // CHECK-SAME: analysisState = "Diagonal"
    %diag_const = arith.constant { metadata = { analysisState = "Diagonal" } } 
      dense<[[2.0, 0.0, 0.0], 
             [0.0, 2.0, 0.0], 
             [0.0, 0.0, 2.0]]> : tensor<3x3xf32>

    %c0 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %empty = tensor.empty() : tensor<3x3xf32>

    // CHECK: scf.for
    %loop_result = scf.for %i = %c0 to %c3 step %c1 
      iter_args(%iter_arg = %sym_const) -> (tensor<3x3xf32>) {
        // CHECK: linalg.add
        // CHECK-SAME: analysisState = "Symmetric"
        %add1 = linalg.add ins(%iter_arg, %sym_const : tensor<3x3xf32>, tensor<3x3xf32>)
                           outs(%empty : tensor<3x3xf32>) 
                           -> tensor<3x3xf32>
        // CHECK: linalg.add
        // CHECK-SAME: analysisState = "Symmetric"
        %add2 = linalg.add ins(%add1, %sym_const : tensor<3x3xf32>, tensor<3x3xf32>)
                           outs(%empty : tensor<3x3xf32>)
                           -> tensor<3x3xf32>
        
        scf.yield %add2 : tensor<3x3xf32>
        // CHECK: analysisState = "Symmetric"
    }

    // CHECK: linalg.mul
    // CHECK-SAME: analysisState = "Diagonal"
    %final_result = linalg.mul ins(%loop_result, %diag_const : tensor<3x3xf32>, tensor<3x3xf32>)
                               outs(%empty : tensor<3x3xf32>)
                               -> tensor<3x3xf32>

    %memref = memref.alloc() : memref<3x3xf32>
    bufferization.materialize_in_destination %final_result in %memref {writable} : (tensor<3x3xf32>, memref<3x3xf32>) -> ()
    %cast = memref.cast %memref : memref<3x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    
    memref.dealloc %memref : memref<3x3xf32>
    return
  }
}
