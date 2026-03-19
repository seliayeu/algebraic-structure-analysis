// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite | FileCheck %s

// CHECK-LABEL: func.func @test_dia_different_bands
func.func @test_dia_different_bands() -> (tensor<3x4xf32>, tensor<3x4xf32>) {
    %diaA = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[1.0, 0.0, 0.0, 0.0], 
               [2.0, 2.0, 0.0, 0.0], 
               [3.0, 3.0, 3.0, 0.0], 
               [0.0, 4.0, 4.0, 4.0]]> : tensor<4x4xf32>
               
    %diaB = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}} 
        dense<[[1.0, 1.0, 0.0, 0.0], 
               [2.0, 2.0, 2.0, 0.0], 
               [0.0, 3.0, 3.0, 3.0], 
               [0.0, 0.0, 4.0, 4.0]]> : tensor<4x4xf32>

    %0 = tensor.empty() : tensor<3x4xf32>
    
    // CHECK: linalg.fill
    // CHECK: %[[ZERO_ADD:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     arith.addf %{{.*}}, %[[ZERO_ADD]]
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     arith.addf %[[ZERO_ADD]], %{{.*}}
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     arith.addf %{{.*}}, %{{.*}}
    // CHECK: %[[ADD_RES:.*]] = scf.for
    // CHECK:   scf.for
    // CHECK:     arith.addf %{{.*}}, %{{.*}}

    %1 = dia.elementwise kind = <add> 
         ins(%diaA, %diaB : tensor<4x4xf32>, tensor<4x4xf32>) 
         outs(%0 : tensor<3x4xf32>) -> tensor<3x4xf32>

    // CHECK: linalg.fill
    // CHECK: %[[ZERO_SUB:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     arith.subf %{{.*}}, %[[ZERO_SUB]]
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     arith.subf %[[ZERO_SUB]], %{{.*}}
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     arith.subf %{{.*}}, %{{.*}}
    // CHECK: %[[SUB_RES:.*]] = scf.for
    // CHECK:   scf.for
    // CHECK:     arith.subf %{{.*}}, %{{.*}}

    %2 = dia.elementwise kind = <sub> 
         ins(%diaA, %diaB : tensor<4x4xf32>, tensor<4x4xf32>) 
         outs(%0 : tensor<3x4xf32>) -> tensor<3x4xf32>

    // CHECK: return %[[ADD_RES]], %[[SUB_RES]]
    return %1, %2 : tensor<3x4xf32>, tensor<3x4xf32>
}
