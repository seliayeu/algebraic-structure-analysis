// RUN: %build/tools/alg-opt %s --banded-analysis | FileCheck %s

module {
  func.func @test_dia_composite_chain() {
    %out = tensor.empty() : tensor<5x5xf32>

    // CHECK: arith.constant
    %A = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 2 : i64, propertyDims = [0, 1], dia = true } } 
             dense<1.0> : tensor<5x5xf32>
             
    // CHECK: arith.constant
    %B = arith.constant { metadata = { upperBw = 1 : i64, lowerBw = 0 : i64, propertyDims = [0, 1], dia = true } } 
             dense<1.0> : tensor<5x5xf32>

    // CHECK: arith.constant
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %C = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 1 : i64, propertyDims = [0, 1], dia = true } } 
             dense<1.0> : tensor<5x5xf32>

    // CHECK: dia.transpose
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %trans = dia.transpose (%A : tensor<5x5xf32>)

    // CHECK: dia.matmul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %matmul = dia.matmul ins(%A, %B : tensor<5x5xf32>, tensor<5x5xf32>) outs(%out : tensor<5x5xf32>) -> tensor<5x5xf32>

    // CHECK: dia.elementwise kind = <mul>
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %mul = dia.elementwise kind = <mul> ins(%matmul, %trans : tensor<5x5xf32>, tensor<5x5xf32>) outs(%out : tensor<5x5xf32>) -> tensor<5x5xf32>

    // CHECK: dia.elementwise kind = <add>
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %add = dia.elementwise kind = <add> ins(%mul, %B : tensor<5x5xf32>, tensor<5x5xf32>) outs(%out : tensor<5x5xf32>) -> tensor<5x5xf32>

    // CHECK: dia.elementwise kind = <mul>
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 0
    %bw_mul = dia.elementwise kind = <mul> ins(%add, %C : tensor<5x5xf32>, tensor<5x5xf32>) outs(%out : tensor<5x5xf32>) -> tensor<5x5xf32>

    return
  }
}
