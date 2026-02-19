// RUN: %build/tools/alg-opt %s --banded-structure-debug | FileCheck %s

module {
  func.func @test_batch_matmul_nonsquare() {
    %out = tensor.empty() : tensor<2x2x3xf32>

    // LHS shape: 2x2x4 (M=2, K=4)
    // RHS shape: 2x4x3 (K=4, N=3)
    // Out shape: 2x2x3 (M=2, N=3) -> Max Lower = 1, Max Upper = 2

    %diag_lhs = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [1, 2] } } 
            dense<[
              [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]],
              [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]]
            ]> : tensor<2x2x4xf32>
    %diag_rhs = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [1, 2] } } 
            dense<[
              [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]],
              [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0],[0.0,0.0,0.0]]
            ]> : tensor<2x4x3xf32>

    %upper_lhs = arith.constant { metadata = { upperBw = 3 : i64, lowerBw = 0 : i64, propertyDims = [1, 2] } } 
             dense<[
               [[1.0,1.0,1.0,1.0],[0.0,1.0,1.0,1.0]],
               [[1.0,1.0,1.0,1.0],[0.0,1.0,1.0,1.0]]
             ]> : tensor<2x2x4xf32>
    %upper_rhs = arith.constant { metadata = { upperBw = 2 : i64, lowerBw = 0 : i64, propertyDims = [1, 2] } } 
             dense<[
               [[1.0,1.0,1.0],[0.0,1.0,1.0],[0.0,0.0,1.0],[0.0,0.0,0.0]],
               [[1.0,1.0,1.0],[0.0,1.0,1.0],[0.0,0.0,1.0],[0.0,0.0,0.0]]
             ]> : tensor<2x4x3xf32>

    %lower_lhs = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 1 : i64, propertyDims = [1, 2] } } 
             dense<[
               [[1.0,0.0,0.0,0.0],[1.0,1.0,0.0,0.0]],
               [[1.0,0.0,0.0,0.0],[1.0,1.0,0.0,0.0]]
             ]> : tensor<2x2x4xf32>
    %lower_rhs = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 3 : i64, propertyDims = [1, 2] } } 
             dense<[
               [[1.0,0.0,0.0],[1.0,1.0,0.0],[1.0,1.0,1.0],[1.0,1.0,1.0]],
               [[1.0,0.0,0.0],[1.0,1.0,0.0],[1.0,1.0,1.0],[1.0,1.0,1.0]]
             ]> : tensor<2x4x3xf32>

    %general_lhs = arith.constant
             dense<[
               [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]],
               [[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]]
             ]> : tensor<2x2x4xf32>


    // (max,max) * (0,0) -> (max,max)
    // CHECK: linalg.batch_matmul
    // CHECK-SAME: lowerBw = 9223372036854775807
    // CHECK-SAME: upperBw = 9223372036854775807
    %res_gen_diag = linalg.batch_matmul ins(%general_lhs, %diag_rhs : tensor<2x2x4xf32>, tensor<2x4x3xf32>) outs(%out : tensor<2x2x3xf32>) -> tensor<2x2x3xf32>

    // (0,0) * (0,0) -> (0,0)
    // CHECK: linalg.batch_matmul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 0
    %res_diag_diag = linalg.batch_matmul ins(%diag_lhs, %diag_rhs : tensor<2x2x4xf32>, tensor<2x4x3xf32>) outs(%out : tensor<2x2x3xf32>) -> tensor<2x2x3xf32>

    // (0,0) * (2,0) -> (2,0)
    // CHECK: linalg.batch_matmul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 2
    %res_diag_up = linalg.batch_matmul ins(%diag_lhs, %upper_rhs : tensor<2x2x4xf32>, tensor<2x4x3xf32>) outs(%out : tensor<2x2x3xf32>) -> tensor<2x2x3xf32>

    // (3,0) * (0,0) -> (3,0) -> Sat(2,0)
    // CHECK: linalg.batch_matmul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 2
    %res_up_diag = linalg.batch_matmul ins(%upper_lhs, %diag_rhs : tensor<2x2x4xf32>, tensor<2x4x3xf32>) outs(%out : tensor<2x2x3xf32>) -> tensor<2x2x3xf32>

    // (3,0) * (2,0) -> (5,0) -> Sat(2,0)
    // CHECK: linalg.batch_matmul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 2
    %res_up_up = linalg.batch_matmul ins(%upper_lhs, %upper_rhs : tensor<2x2x4xf32>, tensor<2x4x3xf32>) outs(%out : tensor<2x2x3xf32>) -> tensor<2x2x3xf32>

    // (3,0) * (0,3) -> (3,3) -> Sat(1,2)
    // CHECK: linalg.batch_matmul
    // CHECK-SAME: lowerBw = 1
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 2
    %res_up_low = linalg.batch_matmul ins(%upper_lhs, %lower_rhs : tensor<2x2x4xf32>, tensor<2x4x3xf32>) outs(%out : tensor<2x2x3xf32>) -> tensor<2x2x3xf32>

    // (0,1) * (0,3) -> (0,4) -> Sat(0,1)
    // CHECK: linalg.batch_matmul
    // CHECK-SAME: lowerBw = 1
    // CHECK-SAME: propertyDims = [1, 2]
    // CHECK-SAME: upperBw = 0
    %res_low_low = linalg.batch_matmul ins(%lower_lhs, %lower_rhs : tensor<2x2x4xf32>, tensor<2x4x3xf32>) outs(%out : tensor<2x2x3xf32>) -> tensor<2x2x3xf32>

    return
  }
}
