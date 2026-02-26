// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite | FileCheck %s

// CHECK: func.func @main() -> tensor<3x4xf32>

// CHECK: tensor.empty() : tensor<3x4xf32>
// CHECK: linalg.fill

// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for

// CHECK: tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<4x4xf32>
// CHECK: tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<4x4xf32>
// CHECK: arith.mulf
// CHECK: arith.addf

// CHECK: arith.subi %{{.*}}, %{{.*}} : index
// CHECK: arith.addi %{{.*}}, %{{.*}} : index
// CHECK: tensor.insert %{{.*}} into %{{.*}}[%{{.*}}, %{{.*}}] : tensor<3x4xf32>

// CHECK: metadata = {layout = "dia", lowerBw = 1 : i64
// CHECK-SAME: upperBw = 1 : i64

// CHECK: return %{{.*}} : tensor<3x4xf32>

func.func @main() -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    %diag = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0,0.0,0.0,0.0],
               [0.0,2.0,0.0,0.0],
               [0.0,0.0,3.0,0.0],
               [0.0,0.0,0.0,4.0]]> : tensor<4x4xf32>
    %tridiag = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 5.0, 0.0, 0.0],
               [8.0, 2.0, 7.0, 0.0],
               [0.0, 9.0, 3.0, 7.0],
               [0.0, 0.0,10.0, 4.0]]> : tensor<4x4xf32>
    %4 = linalg.matmul ins(%diag, %tridiag : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %4 : tensor<4x4xf32>
}
