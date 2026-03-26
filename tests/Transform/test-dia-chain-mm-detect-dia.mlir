// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite | FileCheck %s

module {
  // CHECK-LABEL: func.func @main
  func.func @main() -> f32 {
    
    // --- 1. Hoisted Allocations (Order-independent) ---
    // CHECK-DAG: %[[E_901:.*]] = tensor.empty() {metadata = {lowerBw = 9223372036854775807 : i64, propertyDims = [0, 1], upperBw = 9223372036854775807 : i64}} : tensor<901x1024xf32>
    // CHECK-DAG: %[[E_1024:.*]] = tensor.empty() {metadata = {lowerBw = 9223372036854775807 : i64, propertyDims = [0, 1], upperBw = 9223372036854775807 : i64}} : tensor<1024x1024xf32>
    // CHECK-DAG: %[[E_601:.*]] = tensor.empty() : tensor<601x1024xf32>
    // CHECK-DAG: %[[CST:.*]] = arith.constant {metadata = {{.*}}} dense<1.000000e+00> : tensor<1024x1024xf32>

    %I0 = arith.constant {metadata = {lowerBw = 150 : i64, upperBw = 150 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<1024x1024xf32>
    %I1 = arith.constant {metadata = {lowerBw = 150 : i64, upperBw = 150 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<1024x1024xf32>
    %I2 = arith.constant {metadata = {lowerBw = 150 : i64, upperBw = 150 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<1024x1024xf32>
    %I3 = arith.constant {metadata = {lowerBw = 150 : i64, upperBw = 150 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<1024x1024xf32>

    %c0 = arith.constant 0 : index
    %e0 = tensor.empty(%c0) : tensor<?x1024xf32>
    %e1 = tensor.empty(%c0) : tensor<?x1024xf32>
    %e2 = tensor.empty(%c0) : tensor<?x1024xf32>

    // --- 2. Sequential Logic Blocks ---

    // Stage 1: Produces R1 (601 rows)
    // CHECK: %[[R1:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<601x1024xf32>)
    // CHECK: {metadata = {dia = true, lowerBw = 300 : i64, propertyDims = [0, 1], upperBw = 300 : i64}}
    %R1 = dia.matmul ins(%I0, %I1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                     outs(%e0 : tensor<?x1024xf32>) -> tensor<?x1024xf32>

    // Stage 2: Produces R2 (901 rows)
    // CHECK: %[[R2:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<901x1024xf32>)
    // CHECK: {metadata = {dia = true, lowerBw = 450 : i64, propertyDims = [0, 1], upperBw = 450 : i64}}
    %R2 = dia.matmul ins(%R1, %I2 : tensor<?x1024xf32>, tensor<1024x1024xf32>)
                     outs(%e1 : tensor<?x1024xf32>) -> tensor<?x1024xf32>

    // Stage 3: Produces R3 (1024 rows - Dense)
    // CHECK: %[[R3:.*]] = scf.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (tensor<1024x1024xf32>)
    // CHECK: {metadata = {lowerBw = 600 : i64, propertyDims = [0, 1], upperBw = 600 : i64}}
    %R3 = dia.matmul ins(%R2, %I3 : tensor<?x1024xf32>, tensor<1024x1024xf32>)
                     outs(%e2 : tensor<?x1024xf32>) -> tensor<?x1024xf32>

    %index = arith.constant 4: index
    %result = tensor.extract %R3[%index, %index] : tensor<?x1024xf32>
    return %result : f32
  }
}
