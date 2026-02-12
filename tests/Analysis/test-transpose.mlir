// RUN: %build/tools/alg-opt %s \
// RUN:  --algebraic-structure-debug

module {
  func.func @test_transpose_high_dims() {
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

    %lower_r3 = arith.constant { metadata = { analysisState = "LowerTriangular", propertyDims = [1, 2] } }
                dense<[
                  [[1.0, 0.0], [2.0, 3.0]], 
                  [[4.0, 0.0], [5.0, 6.0]]
                ]> : tensor<2x2x2xf32>

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
    %res_sym_r3 = linalg.transpose
      ins(%sym_r3 : tensor<2x3x3xf32>)
      outs(%sym_r3 : tensor<2x3x3xf32>)
      permutation = [0, 2, 1]

    // UpperTriangular
    // CHECK: analysisState = "LowerTriangular"
    // CHECK-SAME: propertyDims = [1, 2]
    %res_up_r3 = linalg.transpose
      ins(%upper_r3 : tensor<2x3x3xf32>)
      outs(%upper_r3 : tensor<2x3x3xf32>)
      permutation = [0, 2, 1]

    // Diagonal
    // CHECK: analysisState = "Diagonal"
    // CHECK-SAME: propertyDims = [2, 3]
    %res_diag_r4 = linalg.transpose
      ins(%diag_r4 : tensor<2x2x3x3xf32>)
      outs(%diag_r4 : tensor<2x2x3x3xf32>)
      permutation = [0, 1, 3, 2]

    // Identity
    // CHECK: analysisState = "Identity"
    // CHECK-SAME: propertyDims = [2, 3]
    %res_id_r4 = linalg.transpose
      ins(%id_r4 : tensor<2x2x3x3xf32>)
      outs(%id_r4 : tensor<2x2x3x3xf32>)
      permutation = [0, 1, 3, 2]

    // LowerTriangular
    // CHECK: analysisState = "UpperTriangular"
    // CHECK-SAME: propertyDims = [2, 3]
    %res_low_r4 = linalg.transpose
      ins(%low_r4 : tensor<2x2x3x3xf32>)
      outs(%low_r4 : tensor<2x2x3x3xf32>)
      permutation = [0, 1, 3, 2]

    // CHECK: analysisState = "LowerTriangular"
    // CHECK-SAME: propertyDims = [0, 2]
    %res_low_r3 = linalg.transpose
          ins(%lower_r3 : tensor<2x2x2xf32>)
          outs(%lower_r3 : tensor<2x2x2xf32>)
          permutation = [1, 0, 2]
    return
  }
}
