// RUN: %build/tools/alg-opt %s --banded-structure-debug | FileCheck %s

module {
  func.func @test_mul_chain_add() {
    %out = tensor.empty() : tensor<3x3xf32>

    // CHECK: arith.constant
    %lower_A = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 2 : i64, propertyDims = [0, 1] } } 
             dense<[[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]]> : tensor<3x3xf32>
    // CHECK: arith.constant
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %general_B = arith.constant
             dense<[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]> : tensor<3x3xf32>
    %diag_D = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1] } } 
            dense<[[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>


    // C = A * B -> (0,2) * (2,2) -> (0,2)
    // CHECK: linalg.mul
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %res_C = linalg.mul ins(%lower_A, %general_B : tensor<3x3xf32>, tensor<3x3xf32>) 
                        outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // E = B * D -> (2,2) * (0,0) -> (0,0)
    // CHECK: linalg.mul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %res_E = linalg.mul ins(%general_B, %diag_D : tensor<3x3xf32>, tensor<3x3xf32>) 
                        outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // F = C + E -> (0,2) + (0,0) -> (0,2)
    // CHECK: linalg.add
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %res_F = linalg.add ins(%res_C, %res_E : tensor<3x3xf32>, tensor<3x3xf32>) 
                        outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    return
  }
}
