// RUN: %build/tools/alg-opt %s \
// RUN:  --algebraic-structure-debug

module {
  func.func @test_batch_matmul_combinations() {
    %out = tensor.empty() : tensor<2x3x3xf32>

    %sym = arith.constant { metadata = { analysisState = "Symmetric", propertyDims = [1, 2] } } 
           dense<[
             [[2.0,4.0,5.0],[4.0,3.0,9.0],[5.0,9.0,7.0]], 
             [[2.0,4.0,5.0],[4.0,3.0,9.0],[5.0,9.0,7.0]]
           ]> : tensor<2x3x3xf32>

    %diag = arith.constant { metadata = { analysisState = "Diagonal", propertyDims = [1, 2] } } 
            dense<[
              [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]],
              [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]]
            ]> : tensor<2x3x3xf32>

    %id = arith.constant { metadata = { analysisState = "Identity", propertyDims = [1, 2] } } 
          dense<[
            [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
            [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
          ]> : tensor<2x3x3xf32>

    %upper = arith.constant { metadata = { analysisState = "UpperTriangular", propertyDims = [1, 2] } } 
             dense<[
               [[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]],
               [[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]]
             ]> : tensor<2x3x3xf32>

    %lower = arith.constant { metadata = { analysisState = "LowerTriangular", propertyDims = [1, 2] } } 
             dense<[
               [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]],
               [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]]
             ]> : tensor<2x3x3xf32>

    %general = arith.constant { metadata = { analysisState = "General", propertyDims = [1, 2] } } 
             dense<[
               [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]],
               [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]
             ]> : tensor<2x3x3xf32>

    // Symmetric
    // CHECK: analysisState = "General"
    %res_sym_sym = linalg.batch_matmul ins(%sym, %sym : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "General"
    %res_sym_diag = linalg.batch_matmul ins(%sym, %diag : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "Symmetric"
    %res_sym_id = linalg.batch_matmul ins(%sym, %id : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "General"
    %res_sym_up = linalg.batch_matmul ins(%sym, %upper : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "General"
    %res_sym_low = linalg.batch_matmul ins(%sym, %lower : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "General"
    %res_sym_gen = linalg.batch_matmul ins(%sym, %general: tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    // Diagonal
    // CHECK: analysisState = "General"
    %res_diag_sym = linalg.batch_matmul ins(%diag, %sym : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "Diagonal"
    %res_diag_diag = linalg.batch_matmul ins(%diag, %diag : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "Diagonal"
    %res_diag_id = linalg.batch_matmul ins(%diag, %id : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "UpperTriangular"
    %res_diag_up = linalg.batch_matmul ins(%diag, %upper : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "LowerTriangular"
    %res_diag_low = linalg.batch_matmul ins(%diag, %lower : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "General"
    %res_diag_gen = linalg.batch_matmul ins(%diag, %general : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    // Identity
    // CHECK: analysisState = "Symmetric"
    %res_id_sym = linalg.batch_matmul ins(%id, %sym : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "Diagonal"
    %res_id_diag = linalg.batch_matmul ins(%id, %diag : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "Identity"
    %res_id_id = linalg.batch_matmul ins(%id, %id : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "UpperTriangular"
    %res_id_up = linalg.batch_matmul ins(%id, %upper : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "LowerTriangular"
    %res_id_low = linalg.batch_matmul ins(%id, %lower : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "General"
    %res_id_gen = linalg.batch_matmul ins(%id, %general : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    // UpperTriangular
    // CHECK: analysisState = "General"
    %res_up_sym = linalg.batch_matmul ins(%upper, %sym : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "UpperTriangular"
    %res_up_diag = linalg.batch_matmul ins(%upper, %diag : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "UpperTriangular"
    %res_up_id = linalg.batch_matmul ins(%upper, %id : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "UpperTriangular"
    %res_up_up = linalg.batch_matmul ins(%upper, %upper : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "General"
    %res_up_low = linalg.batch_matmul ins(%upper, %lower : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "General"
    %res_up_gen = linalg.batch_matmul ins(%upper, %general : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    // LowerTriangular
    // CHECK: analysisState = "General"
    %res_low_sym = linalg.batch_matmul ins(%lower, %sym : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "LowerTriangular"
    %res_low_diag = linalg.batch_matmul ins(%lower, %diag : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "LowerTriangular"
    %res_low_id = linalg.batch_matmul ins(%lower, %id : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "General"
    %res_low_up = linalg.batch_matmul ins(%lower, %upper : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "LowerTriangular"
    %res_low_low = linalg.batch_matmul ins(%lower, %lower : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    // CHECK: analysisState = "General"
    %res_low_gen = linalg.batch_matmul ins(%lower, %general : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    return
  }
}
