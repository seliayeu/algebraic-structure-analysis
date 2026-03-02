// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite | FileCheck %s

// CHECK-LABEL: func.func @main() -> tensor<3x3xf32>

// CHECK: %[[CST:.*]] = arith.constant
// CHECK-SAME: metadata = {dia = true, lowerBw = 1 : i64, propertyDims = [0, 1], upperBw = 0 : i64}
// CHECK-SAME: tensor<2x3xf32>

// CHECK: linalg.fill
// CHECK-SAME: tensor<3x3xf32>

// outer loop: 0 to 3 (lC+uC+1 = 3)
// CHECK: scf.for %[[I:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<3x3xf32>)

// j loop
// CHECK:   scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<3x3xf32>)

// r loop
// CHECK:     scf.for %[[R:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<3x3xf32>)

// reads and compute
// CHECK:       tensor.extract %{{.*}}[%[[I]], %[[R]]] : tensor<3x3xf32>
// CHECK:       tensor.extract %[[CST]][%[[J]], %[[R]]] : tensor<2x3xf32>
// CHECK:       tensor.extract %[[CST]][%{{.*}}, %{{.*}}] : tensor<2x3xf32>
// CHECK:       arith.mulf
// CHECK:       arith.addf
// CHECK:       tensor.insert %{{.*}} into %{{.*}}[%[[I]], %[[R]]] : tensor<3x3xf32>

// result metadata
// CHECK: metadata = {dia = true, lowerBw = 2 : i64, propertyDims = [0, 1], upperBw = 0 : i64}

// CHECK: return %{{.*}} : tensor<3x3xf32>

module {
  func.func @main() -> tensor<3x3xf32> {
    %0 = tensor.empty() : tensor<3x3xf32>
    %dia1 = arith.constant {metadata = {dia = true, upperBw = 0 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 1.0, 2.0],
               [1.0, 2.0, 3.0]]> : tensor<2x3xf32>
    %dia2 = arith.constant {metadata = {dia = true, upperBw = 0 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 1.0, 2.0],
               [1.0, 2.0, 3.0]]> : tensor<2x3xf32>
    %1 = dia.matmul ins(%dia1, %dia2 : tensor<2x3xf32>, tensor<2x3xf32>)
                    outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
    return %1 : tensor<3x3xf32>
  }
}
