// RUN: %build/tools/alg-opt %s --banded-structure-debug --banded-lowering | %FileCheck %s

module {
  // CHECK-LABEL: func.func @test_matmul
  func.func @test_matmul() -> tensor<3x3xf32> {
    %0 = tensor.empty() : tensor<3x3xf32>
    %diag = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1] } }
            dense<[[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
    %1 = linalg.matmul ins(%diag, %diag : tensor<3x3xf32>, tensor<3x3xf32>)
                       outs(%0: tensor<3x3xf32>) -> tensor<3x3xf32>
    return %1 : tensor<3x3xf32>
  }
}

// CHECK: arith.constant
// CHECK-SAME: lowerBw = 0
// CHECK-SAME: upperBw = 0

// CHECK: tensor.empty
// CHECK-SAME: lowerBw = -1

// CHECK-NOT: linalg.matmul

// CHECK: scf.for
// CHECK-SAME: iter_args

// CHECK: tensor.extract
// CHECK: tensor.extract
// CHECK: arith.mulf
// CHECK: tensor.insert
// CHECK: scf.yield

// CHECK: return
