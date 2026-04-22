// RUN: %build/tools/alg-opt %s --banded-analysis | FileCheck %s

func.func @main () -> tensor<10x10xf32> {
    %0 = arith.constant {metadata={lowerBw = 5 : i64, upperBw = 0 : i64, propertyDims=[0, 1]}} dense<1.0> : tensor<10x10xf32>
    // CHECK: dia.softmax
    // CHECK-SAME: lowerBw = 5
    // CHECK-SAME: upperBw = 0
    // CHECK: return
    %1 = dia.softmax (%0: tensor<10x10xf32>) -> tensor<10x10xf32>
    return %1: tensor<10x10xf32>
}

