// RUN: %build/tools/alg-opt %s --banded-analysis | FileCheck %s

module {
  func.func @test_elementwise_add_combinations() {
    %out = tensor.empty() : tensor<3x3xf32>
    %out_batch = tensor.empty() : tensor<2x3x3xf32>

    %diag = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1] } } 
            dense<[[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
    %upper = arith.constant { metadata = { upperBw = 2 : i64, lowerBw = 0 : i64, propertyDims = [0, 1] } } 
             dense<[[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
    %lower = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 2 : i64, propertyDims = [0, 1] } } 
             dense<[[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]]> : tensor<3x3xf32>
    %general = arith.constant
             dense<[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]> : tensor<3x3xf32>

    %upper_batch = arith.constant { metadata = { upperBw = 2 : i64, lowerBw = 0 : i64, propertyDims = [1, 2] } } 
             dense<[
               [[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]],
               [[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]]
             ]> : tensor<2x3x3xf32>
    %lower_batch = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 2 : i64, propertyDims = [1, 2] } } 
             dense<[
               [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]],
               [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]]
             ]> : tensor<2x3x3xf32>
    %diag_misaligned = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1] } } 
             dense<2.0> : tensor<2x3x3xf32>

    // (0,0) + (0,0) -> (0,0)
    // CHECK: linalg.elementwise
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %res_diag_diag = linalg.elementwise kind=#linalg.elementwise_kind<add> 
      ins(%diag, %diag : tensor<3x3xf32>, tensor<3x3xf32>) 
      outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (0,0) + (2,0) -> (2,0)
    // CHECK: linalg.elementwise
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 2
    %res_diag_up = linalg.elementwise kind=#linalg.elementwise_kind<add> 
      ins(%diag, %upper : tensor<3x3xf32>, tensor<3x3xf32>) 
      outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (2,0) + (2,0) -> (2,0)
    // CHECK: linalg.elementwise
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 2
    %res_up_up = linalg.elementwise kind=#linalg.elementwise_kind<add> 
      ins(%upper, %upper : tensor<3x3xf32>, tensor<3x3xf32>) 
      outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (2,0) + (0,2) -> (2,2)
    // CHECK: linalg.elementwise
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 2
    %res_up_low = linalg.elementwise kind=#linalg.elementwise_kind<add> 
      ins(%upper, %lower : tensor<3x3xf32>, tensor<3x3xf32>) 
      outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (2,2) + (0,0) -> (2,2)
    // CHECK: linalg.elementwise
    // CHECK-SAME: lowerBw = 9223372036854775807
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 9223372036854775807
    %res_gen_diag = linalg.elementwise kind=#linalg.elementwise_kind<add> 
      ins(%general, %diag : tensor<3x3xf32>, tensor<3x3xf32>) 
      outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (2,2) + (0,2) -> (2,2)
    // CHECK: linalg.elementwise
    // CHECK-SAME: lowerBw = 9223372036854775807
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 9223372036854775807
    %res_gen_low = linalg.elementwise kind=#linalg.elementwise_kind<add> 
      ins(%general, %lower : tensor<3x3xf32>, tensor<3x3xf32>) 
      outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (2,0) + (0,2) -> (2,2)
    // CHECK: linalg.elementwise
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 2
    %res_batch_mix = linalg.elementwise kind=#linalg.elementwise_kind<add> 
      ins(%upper_batch, %lower_batch : tensor<2x3x3xf32>, tensor<2x3x3xf32>) 
      outs(%out_batch : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    // (2,0)@[1,2] + (0,0)@[0,1] -> (2,2)@[1,2] (Misalignment forces General)
    // CHECK: linalg.elementwise
    // CHECK-SAME: lowerBw = 9223372036854775807
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 9223372036854775807
    %res_misaligned = linalg.elementwise kind=#linalg.elementwise_kind<add> 
      ins(%upper_batch, %diag_misaligned : tensor<2x3x3xf32>, tensor<2x3x3xf32>) 
      outs(%out_batch : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    return
  }
}
