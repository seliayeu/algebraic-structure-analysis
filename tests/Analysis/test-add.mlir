// RUN: %build/tools/alg-opt %s --banded-structure-debug | FileCheck %s

module {
  func.func @test_add() {
    %out = tensor.empty() : tensor<3x3xf32>
    %out_batch = tensor.empty() : tensor<2x3x3xf32>

    %diag = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1] } } 
            dense<[[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
    %id = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1] } } 
          dense<[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]> : tensor<3x3xf32>
    %upper = arith.constant { metadata = { upperBw = 2 : i64, lowerBw = 0 : i64, propertyDims = [0, 1] } } 
             dense<[[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
    %lower = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 2 : i64, propertyDims = [0, 1] } } 
             dense<[[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]]> : tensor<3x3xf32>
    %general = arith.constant { metadata = { upperBw = 2 : i64, lowerBw = 2 : i64, propertyDims = [0, 1] } } 
             dense<[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]> : tensor<3x3xf32>
    %unmarked = arith.constant dense<[[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]]> : tensor<3x3xf32>

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


    // (0,0) + (0,0) -> (0,0)
    // CHECK: linalg.add
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: upperBw = 0
    %res_diag_diag = linalg.add ins(%diag, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (0,0) + (2,0) -> (2,0)
    // CHECK: linalg.add
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: upperBw = 2
    %res_diag_up = linalg.add ins(%diag, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (2,0) + (0,2) -> (2,2)
    // CHECK: linalg.add
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: upperBw = 2
    %res_up_low = linalg.add ins(%upper, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (2,0) + (2,2) -> (2,2)
    // CHECK: linalg.add
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: upperBw = 2
    %res_up_gen = linalg.add ins(%upper, %general : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (max,max) + (0,0) -> (max,max)
    // CHECK: linalg.add
    // CHECK-SAME: lowerBw = 9223372036854775807
    // CHECK-SAME: upperBw = 9223372036854775807
    %res_unmarked_diag = linalg.add ins(%unmarked, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (2,0) + (0,2) -> (2,2)
    // CHECK: linalg.add
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 2
    %res_batch_mix = linalg.add ins(%upper_batch, %lower_batch : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out_batch : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    // (2,0) + (2,0) -> (2,0)
    // CHECK: linalg.add
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 2
    %res_batch_upper = linalg.add ins(%upper_batch, %upper_batch : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out_batch : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    return
  }
}
