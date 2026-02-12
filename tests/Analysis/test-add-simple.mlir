// RUN: %build/tools/alg-opt %s \
// RUN:  --algebraic-structure-debug

module {
  func.func @test_combinations() {
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
    // CHECK: analysisState = "Symmetric"
    %res_sym_sym = linalg.add ins(%sym, %sym : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "Symmetric"
    %res_sym_diag = linalg.add ins(%sym, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "Symmetric"
    %res_sym_id = linalg.add ins(%sym, %id : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_sym_up = linalg.add ins(%sym, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_sym_low = linalg.add ins(%sym, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_sym_gen = linalg.add ins(%sym, %general: tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // Diagonal
    // CHECK: analysisState = "Symmetric"
    %res_diag_sym = linalg.add ins(%diag, %sym : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "Diagonal"
    %res_diag_diag = linalg.add ins(%diag, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "Diagonal"
    %res_diag_id = linalg.add ins(%diag, %id : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "UpperTriangular"
    %res_diag_up = linalg.add ins(%diag, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "LowerTriangular"
    %res_diag_low = linalg.add ins(%diag, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_diag_gen = linalg.add ins(%diag, %general : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // Identity
    // CHECK: analysisState = "Symmetric"
    %res_id_sym = linalg.add ins(%id, %sym : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "Diagonal"
    %res_id_diag = linalg.add ins(%id, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "Diagonal"
    // (Note: Identity + Identity = 2*Identity, which is Diagonal but not strictly Identity)
    %res_id_id = linalg.add ins(%id, %id : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "UpperTriangular"
    %res_id_up = linalg.add ins(%id, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "LowerTriangular"
    %res_id_low = linalg.add ins(%id, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_id_gen = linalg.add ins(%id, %general : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // UpperTriangular
    // CHECK-NOT: analysisState = "Symmetric"
    %res_up_sym = linalg.add ins(%upper, %sym : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "UpperTriangular"
    %res_up_diag = linalg.add ins(%upper, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "UpperTriangular"
    %res_up_id = linalg.add ins(%upper, %id : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "UpperTriangular"
    %res_up_up = linalg.add ins(%upper, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_up_low = linalg.add ins(%upper, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_up_gen = linalg.add ins(%upper, %general : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // LowerTriangular
    // CHECK: analysisState = "General"
    %res_low_sym = linalg.add ins(%lower, %sym : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "LowerTriangular"
    %res_low_diag = linalg.add ins(%lower, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "LowerTriangular"
    %res_low_id = linalg.add ins(%lower, %id : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_low_up = linalg.add ins(%lower, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "LowerTriangular"
    %res_low_low = linalg.add ins(%lower, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: analysisState = "General"
    %res_low_gen = linalg.add ins(%lower, %general : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    return
  }
}
