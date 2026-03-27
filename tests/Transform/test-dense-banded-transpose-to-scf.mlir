// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite

// CHECK-LABEL: func.func @main
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index
// CHECK: %[[EMPTY:.*]] = tensor.empty()
// CHECK: %[[FILL:.*]] = linalg.fill ins(%{{.*}} : f32) outs(%[[EMPTY]] : tensor<4x4xf32>) -> tensor<4x4xf32>
// CHECK: %[[ILOOP:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[IARG:.*]] = %[[FILL]])
// CHECK:   arith.subi %[[I]], %[[C0]]
// CHECK:   arith.maxsi %{{.*}}, %[[C0]]
// CHECK:   arith.addi %[[I]], %[[C3]]
// CHECK:   arith.minsi %{{.*}}, %[[C4]]
// CHECK:   scf.for %[[J:.*]] = %{{.*}} to %{{.*}} step %[[C1]] iter_args(%[[JARG:.*]] = %[[IARG]])
// CHECK:     tensor.extract %{{.*}}[%[[I]], %[[J]]]
// CHECK:     tensor.insert %{{.*}} into %[[JARG]][%[[J]], %[[I]]]
// CHECK-NOT: linalg.transpose
// CHECK: {metadata = {lowerBw = 3 : i64, propertyDims = [0, 1], upperBw = 0 : i64}}
module {
  func.func @main() -> tensor<4x4xf32> {
    %0 = arith.constant {metadata = {upperBw = 3 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 1.0, 1.0, 1.0],
               [0.0, 2.0, 2.0, 2.0],
               [0.0, 0.0, 3.0, 3.0],
               [0.0, 0.0, 0.0, 4.0]]> : tensor<4x4xf32>
    %1 = tensor.empty() : tensor<4x4xf32>
    %2 = linalg.transpose
        ins(%0 : tensor<4x4xf32>)
        outs(%1 : tensor<4x4xf32>)
        permutation = [1, 0]
    return %2 : tensor<4x4xf32>
  }
}
