// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK:         %[[CST:.*]] = arith.constant {metadata = {lowerBw = 1 : i64, propertyDims = [1, 2], upperBw = 1 : i64}}
// CHECK:         %[[EMPTY:.*]] = tensor.empty() {metadata = {lowerBw = 9223372036854775807 : i64, propertyDims = [0, 1], upperBw = 9223372036854775807 : i64}}
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[C3:.*]] = arith.constant 3 : index
// CHECK:         scf.for %[[B:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%{{.*}} = %[[EMPTY]])
// CHECK:           scf.for %[[I:.*]] = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK:             scf.for {{.*}}
// CHECK:               scf.for {{.*}}
// CHECK:                 tensor.extract %{{.*}}[%[[B]], %{{.*}}, %{{.*}}]
// CHECK:                 tensor.extract %[[CST]][%[[B]], %{{.*}}, %{{.*}}]
// CHECK:                 tensor.extract %[[CST]][%[[B]], %{{.*}}, %{{.*}}]
// CHECK:                 arith.mulf
// CHECK:                 arith.addf
// CHECK:                 tensor.insert %{{.*}} into %{{.*}}[%[[B]], %{{.*}}, %{{.*}}]
// CHECK:         } {metadata = {lowerBw = 2 : i64, propertyDims = [1, 2], upperBw = 2 : i64}}
// CHECK-NOT:     linalg.batch_matmul

module {
  func.func @main() -> tensor<2x3x3xf32> {
    %0 = tensor.empty() : tensor<2x3x3xf32>
    %A = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 1 : i64, propertyDims = [1, 2]}}
            dense<[[[1.0,1.0,0.0],[1.0,2.0,1.0],[0.0,1.0,3.0]],
                   [[1.0,1.0,0.0],[1.0,2.0,1.0],[0.0,1.0,3.0]]]> : tensor<2x3x3xf32>
    %B = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 1 : i64, propertyDims = [1, 2]}}
            dense<[[[1.0,1.0,0.0],[1.0,2.0,1.0],[0.0,1.0,3.0]],
                   [[1.0,1.0,0.0],[1.0,2.0,1.0],[0.0,1.0,3.0]]]> : tensor<2x3x3xf32>
    %C = linalg.batch_matmul
      ins(%A, %B: tensor<2x3x3xf32>, tensor<2x3x3xf32>)
      outs(%0 : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    return %C : tensor<2x3x3xf32>
  }
}
