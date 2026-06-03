// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=false" | FileCheck %s

module {
  func.func @main() {
    // CHECK: arith.constant {metadata = {lowerBw = 1 : i64, propertyDims = [1, 2], upperBw = 0 : i64}}
    %A = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} 
      dense<[[[1.0, 0.0, 0.0, 0.0],
              [2.0, 1.0, 0.0, 0.0],
              [0.0, 2.0, 1.0, 0.0],
              [0.0, 0.0, 2.0, 1.0]],
             [[2.0, 0.0, 0.0, 0.0],
              [4.0, 2.0, 0.0, 0.0],
              [0.0, 4.0, 2.0, 0.0],
              [0.0, 0.0, 4.0, 2.0]]]> : tensor<2x4x4xf32>

    // CHECK: arith.constant {metadata = {lowerBw = 0 : i64, propertyDims = [1, 2], upperBw = 1 : i64}}
    %B = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} 
      dense<[[[1.0, 2.0, 0.0, 0.0],
              [0.0, 1.0, 2.0, 0.0],
              [0.0, 0.0, 1.0, 2.0],
              [0.0, 0.0, 0.0, 1.0]],
             [[2.0, 4.0, 0.0, 0.0],
              [0.0, 2.0, 4.0, 0.0],
              [0.0, 0.0, 2.0, 4.0],
              [0.0, 0.0, 0.0, 2.0]]]> : tensor<2x4x4xf32>

    // CHECK: arith.constant {metadata = {lowerBw = 0 : i64, propertyDims = [1, 2], upperBw = 1 : i64}}
    %C = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} 
      dense<[[[1.0, 1.0, 0.0, 0.0],
              [1.0, 1.0, 1.0, 0.0],
              [0.0, 1.0, 1.0, 1.0],
              [0.0, 0.0, 1.0, 1.0]],
             [[2.0, 2.0, 0.0, 0.0],
              [2.0, 2.0, 2.0, 0.0],
              [0.0, 2.0, 2.0, 2.0],
              [0.0, 0.0, 2.0, 2.0]]]> : tensor<2x4x4xf32>

    // CHECK: arith.constant {metadata = {lowerBw = 0 : i64, propertyDims = [1, 2], upperBw = 0 : i64}}
    %D = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} 
      dense<[[[2.0, 0.0, 0.0, 0.0],
              [0.0, 2.0, 0.0, 0.0],
              [0.0, 0.0, 2.0, 0.0],
              [0.0, 0.0, 0.0, 2.0]],
             [[4.0, 0.0, 0.0, 0.0],
              [0.0, 4.0, 0.0, 0.0],
              [0.0, 0.0, 4.0, 0.0],
              [0.0, 0.0, 0.0, 4.0]]]> : tensor<2x4x4xf32>

    // CHECK: arith.constant {metadata = {lowerBw = 0 : i64, propertyDims = [1, 2], upperBw = 1 : i64}}
    %E = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} 
      dense<[[[1.0, 1.0, 1.0, 0.0],
              [1.0, 1.0, 1.0, 1.0],
              [1.0, 1.0, 1.0, 1.0],
              [0.0, 1.0, 1.0, 1.0]],
             [[2.0, 2.0, 2.0, 0.0],
              [2.0, 2.0, 2.0, 2.0],
              [2.0, 2.0, 2.0, 2.0],
              [0.0, 2.0, 2.0, 2.0]]]> : tensor<2x4x4xf32>

    %empty = tensor.empty() : tensor<2x4x4xf32>

    // CHECK: dia.batch_matmul
    // CHECK-SAME: metadata = {dia = true, lowerBw = 0 : i64, propertyDims = [1, 2], upperBw = 1 : i64}
    %O1 = dia.batch_matmul ins(%A, %B : tensor<2x4x4xf32>, tensor<2x4x4xf32>)
                              outs(%empty : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
    // CHECK: dia.elementwise
    // CHECK-SAME: metadata = {dia = true, lowerBw = 0 : i64, propertyDims = [1, 2], upperBw = 1 : i64}
    %O2 = dia.elementwise kind = <mul> ins(%B, %C : tensor<2x4x4xf32>, tensor<2x4x4xf32>)
                                       outs(%empty : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
    // CHECK: dia.elementwise
    // CHECK-SAME: metadata = {dia = true, lowerBw = 0 : i64, propertyDims = [1, 2], upperBw = 1 : i64}
    %O3 = dia.elementwise kind = <add> ins(%D, %E : tensor<2x4x4xf32>, tensor<2x4x4xf32>)
                                       outs(%empty : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
    // CHECK: dia.elementwise
    // CHECK-SAME: metadata = {dia = true, lowerBw = 0 : i64, propertyDims = [1, 2], upperBw = 1 : i64}
    %O4 = dia.elementwise kind = <mul> ins(%O2, %O3 : tensor<2x4x4xf32>, tensor<2x4x4xf32>)
                                       outs(%empty : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

    // CHECK: dia.elementwise
    // CHECK-SAME: metadata = {dia = true, lowerBw = 0 : i64, propertyDims = [1, 2], upperBw = 1 : i64}
    %O5 = dia.elementwise kind = <mul> ins(%O1, %O4 : tensor<2x4x4xf32>, tensor<2x4x4xf32>)
                                       outs(%empty : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

    return
  }
}
