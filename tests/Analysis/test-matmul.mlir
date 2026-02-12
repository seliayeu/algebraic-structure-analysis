// RUN: %build/tools/alg-opt %s \
// RUN:  --algebraic-structure-debug

module {
  func.func @test_matmul_combinations() {
    %out = tensor.empty() : tensor<3x3xf32>

    %sym = arith.constant { metadata = { analysisState = "Symmetric", propertyDims = [0, 1] } } 
           dense<[[2.0,4.0,5.0],[4.0,3.0,9.0],[5.0,9.0,7.0]]> : tensor<3x3xf32>
    %diag = arith.constant { metadata = { analysisState = "Diagonal", propertyDims = [0, 1] } } 
            dense<[[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
    %id = arith.constant { metadata = { analysisState = "Identity", propertyDims = [0, 1] } } 
          dense<[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]> : tensor<3x3xf32>
    %upper = arith.constant { metadata = { analysisState = "UpperTriangular", propertyDims = [0, 1] } } 
             dense<[[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
    %lower = arith.constant { metadata = { analysisState = "LowerTriangular", propertyDims = [0, 1] } } 
             dense<[[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]]> : tensor<3x3xf32>
    %general = arith.constant { metadata = { analysisState = "General", propertyDims = [0, 1] } } 
             dense<[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]> : tensor<3x3xf32>

    // Symmetric
    // CHECK: analysisState = "General"
    %res_sym_sym = linalg.matmul ins(%sym, %sym : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_sym_diag = linalg.matmul ins(%sym, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "Symmetric"
    %res_sym_id = linalg.matmul ins(%sym, %id : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_sym_up = linalg.matmul ins(%sym, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_sym_low = linalg.matmul ins(%sym, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_sym_gen = linalg.matmul ins(%sym, %general: tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // Diagonal
    // CHECK: analysisState = "General"
    %res_diag_sym = linalg.matmul ins(%diag, %sym : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "Diagonal"
    %res_diag_diag = linalg.matmul ins(%diag, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "Diagonal"
    %res_diag_id = linalg.matmul ins(%diag, %id : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "UpperTriangular"
    %res_diag_up = linalg.matmul ins(%diag, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "LowerTriangular"
    %res_diag_low = linalg.matmul ins(%diag, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_diag_gen = linalg.matmul ins(%diag, %general : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // Identity
    // CHECK: analysisState = "Symmetric"
    %res_id_sym = linalg.matmul ins(%id, %sym : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "Diagonal"
    %res_id_diag = linalg.matmul ins(%id, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "Identity"
    %res_id_id = linalg.matmul ins(%id, %id : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "UpperTriangular"
    %res_id_up = linalg.matmul ins(%id, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "LowerTriangular"
    %res_id_low = linalg.matmul ins(%id, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_id_gen = linalg.matmul ins(%id, %general : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // UpperTriangular
    // CHECK: analysisState = "General"
    %res_up_sym = linalg.matmul ins(%upper, %sym : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "UpperTriangular"
    %res_up_diag = linalg.matmul ins(%upper, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "UpperTriangular"
    %res_up_id = linalg.matmul ins(%upper, %id : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "UpperTriangular"
    %res_up_up = linalg.matmul ins(%upper, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_up_low = linalg.matmul ins(%upper, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_up_gen = linalg.matmul ins(%upper, %general : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // LowerTriangular
    // CHECK: analysisState = "General"
    %res_low_sym = linalg.matmul ins(%lower, %sym : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "LowerTriangular"
    %res_low_diag = linalg.matmul ins(%lower, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "LowerTriangular"
    %res_low_id = linalg.matmul ins(%lower, %id : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_low_up = linalg.matmul ins(%lower, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "LowerTriangular"
    %res_low_low = linalg.matmul ins(%lower, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_low_gen = linalg.matmul ins(%lower, %general : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    return
  }
}
