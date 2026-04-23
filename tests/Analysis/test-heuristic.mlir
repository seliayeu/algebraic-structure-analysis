// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-analysis="heuristic=true"| FileCheck %s

module {
  // CHECK-LABEL: func.func @insert_from_dense_lhs
  // CHECK:       %[[FROM:.*]] = dia.from_dense %{{.*}} {metadata = {dia = true, lowerBw = 1 : i64, propertyDims = [0, 1], upperBw = 1 : i64}}
  // CHECK-SAME:      : tensor<4x4xf32> -> tensor<3x4xf32>
  // CHECK:       dia.matmul ins(%[[FROM]], %{{.*}} : tensor<3x4xf32>, tensor<1x4xf32>)
  // CHECK-NOT:   dia.from_dense
  func.func @insert_from_dense_lhs() -> tensor<3x4xf32> {
    %dense_lhs = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 0.0, 0.0],
               [3.0, 4.0, 5.0, 0.0],
               [0.0, 6.0, 7.0, 8.0],
               [0.0, 0.0, 9.0, 1.0]]> : tensor<4x4xf32>

    %dia_rhs = arith.constant {metadata = {dia = true, upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>

    %out = tensor.empty() : tensor<3x4xf32>

    %result = dia.matmul ins(%dense_lhs, %dia_rhs : tensor<4x4xf32>, tensor<1x4xf32>)
                         outs(%out : tensor<3x4xf32>)
                         -> tensor<3x4xf32>
    return %result : tensor<3x4xf32>
  }

  // CHECK-LABEL: func.func @insert_from_dense_both
  // CHECK-DAG:   %[[FROM_LHS:.*]] = dia.from_dense %{{.*}} {metadata = {dia = true, lowerBw = 1 : i64, propertyDims = [0, 1], upperBw = 1 : i64}}
  // CHECK-DAG:   %[[FROM_RHS:.*]] = dia.from_dense %{{.*}} {metadata = {dia = true, lowerBw = 1 : i64, propertyDims = [0, 1], upperBw = 1 : i64}}
  // CHECK:       dia.matmul ins(%[[FROM_LHS]], %[[FROM_RHS]] : tensor<3x4xf32>, tensor<3x4xf32>)
  func.func @insert_from_dense_both() -> tensor<5x4xf32> {
    %dense_lhs = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 0.0, 0.0],
               [3.0, 4.0, 5.0, 0.0],
               [0.0, 6.0, 7.0, 8.0],
               [0.0, 0.0, 9.0, 1.0]]> : tensor<4x4xf32>

    %dense_rhs = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 0.0, 0.0],
               [3.0, 4.0, 5.0, 0.0],
               [0.0, 6.0, 7.0, 8.0],
               [0.0, 0.0, 9.0, 1.0]]> : tensor<4x4xf32>

    %out = tensor.empty() : tensor<5x4xf32>

    %result = dia.matmul ins(%dense_lhs, %dense_rhs : tensor<4x4xf32>, tensor<4x4xf32>)
                         outs(%out : tensor<5x4xf32>)
                         -> tensor<5x4xf32>
    return %result : tensor<5x4xf32>
  }

  // CHECK-LABEL: func.func @no_from_dense_when_already_dia
  // CHECK-NOT:   dia.from_dense
  // CHECK:       dia.matmul
  func.func @no_from_dense_when_already_dia() -> tensor<3x4xf32> {
    %dia_lhs = arith.constant {metadata = {dia = true, upperBw = 1 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 5.0, 7.0, 8.0],
               [1.0, 4.0, 7.0, 0.0],
               [3.0, 6.0, 0.0, 0.0]]> : tensor<3x4xf32>

    %dia_rhs = arith.constant {metadata = {dia = true, upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>

    %out = tensor.empty() : tensor<3x4xf32>

    %result = dia.matmul ins(%dia_lhs, %dia_rhs : tensor<3x4xf32>, tensor<1x4xf32>)
                         outs(%out : tensor<3x4xf32>)
                         -> tensor<3x4xf32>
    return %result : tensor<3x4xf32>
  }
}
