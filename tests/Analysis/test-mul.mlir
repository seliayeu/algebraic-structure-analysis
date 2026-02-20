// RUN: %build/tools/alg-opt %s --banded-analysis | FileCheck %s

module {
  func.func @test_mul() {
    %out = tensor.empty() : tensor<3x3xf32>

    %diag = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1] } } 
            dense<[[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
    %upper = arith.constant { metadata = { upperBw = 2 : i64, lowerBw = 0 : i64, propertyDims = [0, 1] } } 
             dense<[[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
    %lower = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 2 : i64, propertyDims = [0, 1] } } 
             dense<[[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]]> : tensor<3x3xf32>
    %general = arith.constant
             dense<[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]> : tensor<3x3xf32>

    // (0,0) * (0,0) -> (0,0)
    // CHECK: linalg.mul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %res_diag_diag = linalg.mul ins(%diag, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (0,0) * (2,0) -> (0,0)
    // CHECK: linalg.mul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %res_diag_up = linalg.mul ins(%diag, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (2,0) * (2,0) -> (2,0)
    // CHECK: linalg.mul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 2
    %res_up_up = linalg.mul ins(%upper, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (2,0) * (0,2) -> (0,0)
    // CHECK: linalg.mul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %res_up_low = linalg.mul ins(%upper, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (2,2) * (0,0) -> (0,0)
    // CHECK: linalg.mul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %res_gen_diag = linalg.mul ins(%general, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (2,2) * (0,2) -> (0,2)
    // CHECK: linalg.mul
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %res_gen_low = linalg.mul ins(%general, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    return
  }
}
