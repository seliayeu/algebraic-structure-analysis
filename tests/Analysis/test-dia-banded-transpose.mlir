// RUN: %build/tools/alg-opt %s --banded-analysis | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK: arith.constant {metadata = {lowerBw = 2 : i64, propertyDims = [0, 1], upperBw = 0 : i64}}
// CHECK: dia.transpose({{.*}} {metadata = {dia = true, lowerBw = 0 : i64, propertyDims = [0, 1], upperBw = 2 : i64}}
// CHECK-NOT: lowerBw = 0 : i64{{.*}}upperBw = 2 : i64{{.*}}dia.transpose
module {
  func.func @main() -> tensor<5x3xf32> {
    %dia = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 2 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 0.0, 0.7],
               [0.0, 4.0, 8.0],
               [1.0, 5.0, 9.0],
               [2.0, 6.0, 0.0],
               [3.0, 0.0, 0.0]]> : tensor<5x3xf32>
    %transposed = dia.transpose (%dia: tensor<5x3xf32>)
    return %transposed : tensor<5x3xf32>
  }
}
