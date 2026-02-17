// RUN: %build/tools/alg-opt %s --banded-structure-debug | FileCheck %s

module {
  func.func @test_batch_matmul() {
    %out = tensor.empty() : tensor<2x3x3xf32>

    %diag = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [1, 2] } } 
            dense<[
              [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]],
              [[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]]
            ]> : tensor<2x3x3xf32>
    %upper = arith.constant { metadata = { upperBw = 2 : i64, lowerBw = 0 : i64, propertyDims = [1, 2] } } 
             dense<[
               [[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]],
               [[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]]
             ]> : tensor<2x3x3xf32>
    %lower = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 2 : i64, propertyDims = [1, 2] } } 
             dense<[
               [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]],
               [[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]]
             ]> : tensor<2x3x3xf32>
    %general = arith.constant
             dense<[
               [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]],
               [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]
             ]> : tensor<2x3x3xf32>

    // (max,max) * (0,0) -> (max,max)
    // dimensions don't matter for general so this output is valid
    // CHECK: linalg.batch_matmul
    // CHECK-SAME: lowerBw = 9223372036854775807
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 9223372036854775807
    %res_gen_diag = linalg.batch_matmul ins(%general, %diag : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    
    // (0,0) * (0,0) -> (0,0)
    // CHECK: linalg.batch_matmul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 0
    %res_diag_diag = linalg.batch_matmul ins(%diag, %diag : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    
    // (0,0) * (2,0) -> (2,0)
    // CHECK: linalg.batch_matmul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 2
    %res_diag_up = linalg.batch_matmul ins(%diag, %upper : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    
    // (2,0) * (0,0) -> (2,0)
    // CHECK: linalg.batch_matmul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 2
    %res_up_diag = linalg.batch_matmul ins(%upper, %diag : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    
    // (2,0) * (2,0) -> (2,0)
    // CHECK: linalg.batch_matmul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 2
    %res_up_up = linalg.batch_matmul ins(%upper, %upper : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    
    // (2,0) * (0,2) -> (2,2)
    // CHECK: linalg.batch_matmul
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 2
    %res_up_low = linalg.batch_matmul ins(%upper, %lower : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    
    // (0,2) * (0,2) -> (0,2)
    // CHECK: linalg.batch_matmul
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 0
    %res_low_low = linalg.batch_matmul ins(%lower, %lower : tensor<2x3x3xf32>, tensor<2x3x3xf32>) outs(%out : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

    return
  }
}
