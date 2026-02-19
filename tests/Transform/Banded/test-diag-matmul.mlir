// RUN: %build/tools/alg-opt %s --banded-structure-debug | FileCheck %s

module {
  func.func @test_matmul() -> tensor<3x3xf32> {
    %0 = tensor.empty() : tensor<3x3xf32>

    %diag = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1] } } 
            dense<[[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>


    // (0,0) * (0,0) -> (0,0)
    // CHECK: linalg.matmul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: upperBw = 0
    %1 = linalg.matmul ins(%diag, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%0: tensor<3x3xf32>) -> tensor<3x3xf32>

    return %1 : tensor<3x3xf32>
  }
}

