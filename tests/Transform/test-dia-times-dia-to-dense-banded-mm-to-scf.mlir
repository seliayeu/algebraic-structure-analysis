// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[C3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[C0_I64:.*]] = arith.constant 0 : i64
// CHECK: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<3x3xf32>) -> tensor<3x3xf32>
// CHECK: scf.for %[[ROW:.*]] = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK:   scf.for %[[COL:.*]] = %[[C0]] to %[[C3]] step %[[C1]]
// CHECK:     arith.maxsi
// CHECK:     arith.minsi
// CHECK:     scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %[[C1]]
// CHECK:       arith.index_cast %[[ROW]] : index to i64
// CHECK:       arith.index_cast %[[K]] : index to i64
// CHECK:       arith.index_cast %[[COL]] : index to i64
// CHECK:       arith.addi %{{.*}}, %[[C1_I64]]
// CHECK:       arith.addi %{{.*}}, %[[C0_I64]]
// CHECK:       tensor.extract %{{.*}}[%{{.*}}, %[[ROW]]]
// CHECK:       tensor.extract %{{.*}}[%{{.*}}, %[[K]]]
// CHECK:       tensor.extract %{{.*}}[%[[ROW]], %[[COL]]]
// CHECK:       arith.mulf
// CHECK:       arith.addf
// CHECK:       tensor.insert %{{.*}} into %{{.*}}[%[[ROW]], %[[COL]]]
// CHECK-NOT: dia.matmul
module {
  func.func @main() -> tensor<3x3xf32> {
    %0 = tensor.empty() : tensor<3x3xf32>
    %diaA = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 4.0, 8.0],
               [1.0, 5.0, 9.0]]> : tensor<2x3xf32>
    %diaB = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 5.0, 9.0],
               [2.0, 6.0, 0.0]]> : tensor<2x3xf32>
    %1 = dia.matmul ins(%diaA, %diaB: tensor<2x3xf32>, tensor<2x3xf32>)
                    outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
    return %1 : tensor<3x3xf32>
  }
}
