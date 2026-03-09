// RUN: %build/tools/alg-opt %s --banded-analysis | FileCheck %s
module {
  func.func @test_mul_chain_add() {
    %out = tensor.empty() : tensor<3x3xf32>
    // CHECK: arith.constant
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %lower_A = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 2 : i64, propertyDims = [0, 1] } }
             dense<[[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]]> : tensor<3x3xf32>
    // CHECK: arith.constant
    %general_B = arith.constant
             dense<[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]> : tensor<3x3xf32>
    // CHECK: arith.constant
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %diag_D = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1] } }
            dense<[[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
    // C = A * B -> B(2,0) * B(MAX,MAX) -> B(2,0)
    // CHECK: linalg.mul
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %res_C = linalg.mul ins(%lower_A, %general_B : tensor<3x3xf32>, tensor<3x3xf32>)
                        outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // E = B * D -> B(MAX,MAX) * B(0,0) -> B(0,0)
    // CHECK: linalg.mul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %res_E = linalg.mul ins(%general_B, %diag_D : tensor<3x3xf32>, tensor<3x3xf32>)
                        outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    // F = C + E -> B(2,0) + B(0,0) -> B(2,0)
    // CHECK: linalg.add
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %res_F = linalg.add ins(%res_C, %res_E : tensor<3x3xf32>, tensor<3x3xf32>)
                        outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    return
  }
}
