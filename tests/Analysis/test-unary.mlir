// RUN: %build/tools/alg-opt %s --banded-analysis | FileCheck %s

module {
  func.func @test_unary_elementwise() {
    %out_r3 = tensor.empty() : tensor<2x3x3xf32>
    %out_r4 = tensor.empty() : tensor<2x2x3x3xf32>

    %general_r3 = arith.constant { metadata = { upperBw = 2 : i64, lowerBw = 2 : i64, propertyDims = [1, 2] } } 
              dense<[
                [[2.0,4.0,5.0],[4.0,3.0,9.0],[5.0,9.0,7.0]], 
                [[2.0,4.0,5.0],[4.0,3.0,9.0],[5.0,9.0,7.0]]
              ]> : tensor<2x3x3xf32>
    %upper_r3 = arith.constant { metadata = { upperBw = 2 : i64, lowerBw = 0 : i64, propertyDims = [1, 2] } } 
                dense<[
                  [[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]],
                  [[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]]
                ]> : tensor<2x3x3xf32>
    %diag_r4 = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [2, 3] } } 
               dense<[
                 [ [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]], [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]] ],
                 [ [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]], [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]] ]
               ]> : tensor<2x2x3x3xf32>
    %low_r4 = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 2 : i64, propertyDims = [2, 3] } } 
              dense<[
                [ [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]], [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]] ],
                [ [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]], [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]] ]
              ]> : tensor<2x2x3x3xf32>

    // (2,2) -> (2,2)
    // CHECK: linalg.elementwise
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 2
    %res_gen_r3 = linalg.elementwise kind = #linalg.elementwise_kind<square>
        ins(%general_r3 : tensor<2x3x3xf32>) 
        outs(%out_r3 : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    // (2,0) -> (2,0)
    // CHECK: linalg.elementwise
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 2
    %res_up_r3 = linalg.elementwise kind = #linalg.elementwise_kind<square>
        ins(%upper_r3 : tensor<2x3x3xf32>) 
        outs(%out_r3 : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    // (0,0) -> (0,0)
    // CHECK: linalg.elementwise
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [2, 3]
    // CHECK-SAME: upperBw = 0
    %res_diag_r4 = linalg.elementwise kind = #linalg.elementwise_kind<square>
        ins(%diag_r4 : tensor<2x2x3x3xf32>) 
        outs(%out_r4 : tensor<2x2x3x3xf32>) -> tensor<2x2x3x3xf32>

    // (0,2) -> (0,2)
    // CHECK: linalg.elementwise
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [2, 3]
    // CHECK-SAME: upperBw = 0
    %res_low_r4 = linalg.elementwise kind = #linalg.elementwise_kind<square>
        ins(%low_r4 : tensor<2x2x3x3xf32>) 
        outs(%out_r4 : tensor<2x2x3x3xf32>) -> tensor<2x2x3x3xf32>

    return
  }
}
