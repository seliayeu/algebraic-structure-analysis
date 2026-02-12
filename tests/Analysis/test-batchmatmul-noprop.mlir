// RUN: %build/tools/alg-opt %s \
// RUN:  --algebraic-structure-debug

module {
  func.func @test_batch_matmul_misaligned_dims() {
    %out = tensor.empty() : tensor<2x3x3xf32>
    
    // Symmetric
    %sym_misaligned = arith.constant { 
      metadata = { 
        analysisState = "Symmetric", 
        propertyDims = [0, 1]
      } 
    } dense<[
      [[2.0,4.0,5.0],[4.0,3.0,9.0],[5.0,9.0,7.0]], 
      [[2.0,4.0,5.0],[4.0,3.0,9.0],[5.0,9.0,7.0]]
    ]> : tensor<2x3x3xf32>

    // Identity
    %id_misaligned = arith.constant { 
      metadata = { 
        analysisState = "Identity", 
        propertyDims = [0, 1] 
      } 
    } dense<[
      [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
      [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
    ]> : tensor<2x3x3xf32>

    // Symmetric
    // CHECK: analysisState = "General"
    %res_sym_bad = linalg.batch_matmul 
      ins(%sym_misaligned, %sym_misaligned : tensor<2x3x3xf32>, tensor<2x3x3xf32>) 
      outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    // Identity
    // CHECK: analysisState = "General"
    %res_id_bad = linalg.batch_matmul 
      ins(%id_misaligned, %id_misaligned : tensor<2x3x3xf32>, tensor<2x3x3xf32>) 
      outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    return
  }
}
