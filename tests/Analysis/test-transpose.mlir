// RUN: %build/tools/alg-opt %s --banded-analysis | FileCheck %s

module {
  func.func @test_transpose() {
    
    %sym_r3 = arith.constant { metadata = { upperBw = 2 : i64, lowerBw = 2 : i64, propertyDims = [1, 2] } } 
              dense<[
                [[2.0,4.0,5.0],[4.0,3.0,9.0],[5.0,9.0,7.0]], 
                [[2.0,4.0,5.0],[4.0,3.0,9.0],[5.0,9.0,7.0]]
              ]> : tensor<2x3x3xf32>
    %upper_r3 = arith.constant { metadata = { upperBw = 2 : i64, lowerBw = 0 : i64, propertyDims = [1, 2] } } 
                dense<[
                  [[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]],
                  [[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]]
                ]> : tensor<2x3x3xf32>
    %lower_r3 = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 1 : i64, propertyDims = [1, 2] } }
                dense<[
                  [[1.0, 0.0], [2.0, 3.0]], 
                  [[4.0, 0.0], [5.0, 6.0]]
                ]> : tensor<2x2x2xf32>
    %diag_r4 = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [2, 3] } } 
               dense<[
                 [ [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]], [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]] ],
                 [ [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]], [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]] ]
               ]> : tensor<2x2x3x3xf32>
    %id_r4 = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [2, 3] } } 
             dense<[
               [ [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]], [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]] ],
               [ [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]], [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]] ]
             ]> : tensor<2x2x3x3xf32>
    %low_r4 = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 2 : i64, propertyDims = [2, 3] } } 
              dense<[
                [ [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]], [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]] ],
                [ [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]], [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]] ]
              ]> : tensor<2x2x3x3xf32>

    // CHECK: linalg.transpose
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 2
    %res_sym_r3 = linalg.transpose
      ins(%sym_r3 : tensor<2x3x3xf32>)
      outs(%sym_r3 : tensor<2x3x3xf32>)
      permutation = [0, 2, 1]

    // CHECK: linalg.transpose
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 0
    %res_up_r3 = linalg.transpose
      ins(%upper_r3 : tensor<2x3x3xf32>)
      outs(%upper_r3 : tensor<2x3x3xf32>)
      permutation = [0, 2, 1]
    
    // CHECK: linalg.transpose
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [2, 3]
    // CHECK-SAME: upperBw = 0
    %res_diag_r4 = linalg.transpose
      ins(%diag_r4 : tensor<2x2x3x3xf32>)
      outs(%diag_r4 : tensor<2x2x3x3xf32>)
      permutation = [0, 1, 3, 2]

    // CHECK: linalg.transpose
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [2, 3]
    // CHECK-SAME: upperBw = 0
    %res_id_r4 = linalg.transpose
      ins(%id_r4 : tensor<2x2x3x3xf32>)
      outs(%id_r4 : tensor<2x2x3x3xf32>)
      permutation = [0, 1, 3, 2]

    // CHECK: linalg.transpose
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [2, 3]
    // CHECK-SAME: upperBw = 2
    %res_low_r4 = linalg.transpose
      ins(%low_r4 : tensor<2x2x3x3xf32>)
      outs(%low_r4 : tensor<2x2x3x3xf32>)
      permutation = [0, 1, 3, 2]

    // CHECK: linalg.transpose
    // CHECK-SAME: lowerBw = 1
    // CHECK-SAME: propertyDims = [0, 2]
    // CHECK-SAME: upperBw = 0
    %res_low_r3 = linalg.transpose
          ins(%lower_r3 : tensor<2x2x2xf32>)
          outs(%lower_r3 : tensor<2x2x2xf32>)
          permutation = [1, 0, 2]
    return
  }
}
