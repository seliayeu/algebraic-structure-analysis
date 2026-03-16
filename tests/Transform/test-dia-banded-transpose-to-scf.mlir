// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[C5:.*]] = arith.constant 5 : index
// CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<5x3xf32>
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[EMPTY]] : tensor<5x3xf32>) -> tensor<5x3xf32>
// CHECK: %[[ILOOP:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C5]] step %[[C1]] iter_args(%[[IARG:.*]] = %[[FILL]])
// CHECK:   arith.subi %[[C2]], %[[I]]
// CHECK:   arith.subi %[[I]], %[[C2]]
// CHECK:   scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step %[[C1]] iter_args(%[[JARG:.*]] = %[[IARG]])
// CHECK:     tensor.extract %{{.*}}[%[[I]], %[[J]]]
// CHECK:     tensor.insert %{{.*}} into %[[JARG]][%{{.*}}, %{{.*}}]
// CHECK-NOT: dia.transpose
module {
  func.func @main() -> tensor<5x3xf32> {
    %dia = arith.constant {metadata = {upperBw = 2 : i64, lowerBw = 2 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 0.0, 0.7],
               [0.0, 4.0, 8.0],
               [1.0, 5.0, 9.0],
               [2.0, 6.0, 0.0],
               [3.0, 0.0, 0.0]]> : tensor<5x3xf32>
    %transposed = dia.transpose (%dia: tensor<5x3xf32>)
    return %transposed : tensor<5x3xf32>
  }
}
