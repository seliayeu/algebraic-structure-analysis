// RUN: %build/tools/alg-opt %s \
// RUN:  --algebraic-structure-debug

module {
  func.func @test_elementwise_high_dims_data() {
    %out_r3 = tensor.empty() : tensor<2x3x3xf32>
    %out_r4 = tensor.empty() : tensor<2x2x3x3xf32>

    %sym_r3 = arith.constant { metadata = { analysisState = "Symmetric", propertyDims = [1, 2] } } 
              dense<[
                [[2.0,4.0,5.0],[4.0,3.0,9.0],[5.0,9.0,7.0]], 
                [[2.0,4.0,5.0],[4.0,3.0,9.0],[5.0,9.0,7.0]]
              ]> : tensor<2x3x3xf32>

    %upper_r3 = arith.constant { metadata = { analysisState = "UpperTriangular", propertyDims = [1, 2] } } 
                dense<[
                  [[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]],
                  [[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]]
                ]> : tensor<2x3x3xf32>
    %diag_r4 = arith.constant { metadata = { analysisState = "Diagonal", propertyDims = [2, 3] } } 
               dense<[
                 [ [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]], [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]] ],
                 [ [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]], [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]] ]
               ]> : tensor<2x2x3x3xf32>
    %id_r4 = arith.constant { metadata = { analysisState = "Identity", propertyDims = [2, 3] } } 
             dense<[
               [ [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]], [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]] ],
               [ [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]], [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]] ]
             ]> : tensor<2x2x3x3xf32>
    %low_r4 = arith.constant { metadata = { analysisState = "LowerTriangular", propertyDims = [2, 3] } } 
              dense<[
                [ [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]], [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]] ],
                [ [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]], [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]] ]
              ]> : tensor<2x2x3x3xf32>

    // Symmetric
    // CHECK: analysisState = "Symmetric"
    // CHECK-SAME: propertyDims = [1, 2]
    %res_sym_r3 = linalg.elementwise kind = #linalg.elementwise_kind<square>
        ins(%sym_r3 : tensor<2x3x3xf32>) 
        outs(%out_r3 : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    // UpperTriangular
    // CHECK: analysisState = "UpperTriangular"
    // CHECK-SAME: propertyDims = [1, 2]
    %res_up_r3 = linalg.elementwise kind = #linalg.elementwise_kind<square>
        ins(%upper_r3 : tensor<2x3x3xf32>) 
        outs(%out_r3 : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    // Diagonal
    // CHECK: analysisState = "Diagonal"
    // CHECK-SAME: propertyDims = [2, 3]
    %res_diag_r4 = linalg.elementwise kind = #linalg.elementwise_kind<square>
        ins(%diag_r4 : tensor<2x2x3x3xf32>) 
        outs(%out_r4 : tensor<2x2x3x3xf32>) -> tensor<2x2x3x3xf32>

    // Identity
    // CHECK: analysisState = "Diagonal"
    // CHECK-SAME: propertyDims = [2, 3]
    %res_id_r4 = linalg.elementwise kind = #linalg.elementwise_kind<square>
        ins(%id_r4 : tensor<2x2x3x3xf32>) 
        outs(%out_r4 : tensor<2x2x3x3xf32>) -> tensor<2x2x3x3xf32>

    // LowerTriangular
    // CHECK: analysisState = "LowerTriangular"
    // CHECK-SAME: propertyDims = [2, 3]
    %res_low_r4 = linalg.elementwise kind = #linalg.elementwise_kind<square>
        ins(%low_r4 : tensor<2x2x3x3xf32>) 
        outs(%out_r4 : tensor<2x2x3x3xf32>) -> tensor<2x2x3x3xf32>

    return
  }
}
