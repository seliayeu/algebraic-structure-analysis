// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite | FileCheck %s

// CHECK-LABEL: func.func @test_dia_different_bands
func.func @test_dia_different_bands() -> (tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>) {
    %diaA = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}} 
        dense<[[1.0, 1.0, 1.0, 0.0], [2.0, 2.0, 2.0, 2.0]]> : tensor<2x4xf32>
    %diaB = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[3.0, 3.0, 3.0, 3.0], [0.0, 4.0, 4.0, 4.0]]> : tensor<2x4xf32>

    %0 = tensor.empty() : tensor<3x4xf32>
    
    // CHECK: linalg.fill
    // CHECK: %[[ZERO_ADD:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     arith.addf %{{.*}}, %[[ZERO_ADD]]
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     arith.addf %[[ZERO_ADD]], %{{.*}}
    // CHECK: %[[ADD_RES:.*]] = scf.for
    // CHECK:   scf.for
    // CHECK:     arith.addf %{{.*}}, %{{.*}}

    %1 = dia.elementwise kind = <add> 
         ins(%diaA, %diaB : tensor<2x4xf32>, tensor<2x4xf32>) 
         outs(%0 : tensor<3x4xf32>) -> tensor<3x4xf32>

    // CHECK: linalg.fill
    // CHECK: %[[ZERO_SUB:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     arith.subf %{{.*}}, %[[ZERO_SUB]]
    // CHECK: scf.for
    // CHECK:   scf.for
    // CHECK:     arith.subf %[[ZERO_SUB]], %{{.*}}
    // CHECK: %[[SUB_RES:.*]] = scf.for
    // CHECK:   scf.for
    // CHECK:     arith.subf %{{.*}}, %{{.*}}

    %2 = dia.elementwise kind = <sub> 
         ins(%diaA, %diaB : tensor<2x4xf32>, tensor<2x4xf32>) 
         outs(%0 : tensor<3x4xf32>) -> tensor<3x4xf32>

    // CHECK: linalg.fill
    // CHECK: %[[MUL_RES:.*]] = scf.for
    // CHECK:   scf.for
    // CHECK:     arith.mulf %{{.*}}, %{{.*}}
    // CHECK-NOT: arith.mulf

    %3 = dia.elementwise kind = <mul> 
         ins(%diaA, %diaB : tensor<2x4xf32>, tensor<2x4xf32>) 
         outs(%0 : tensor<3x4xf32>) -> tensor<3x4xf32>

    // CHECK: return %[[ADD_RES]], %[[SUB_RES]], %[[MUL_RES]]
    return %1, %2, %3 : tensor<3x4xf32>, tensor<3x4xf32>, tensor<3x4xf32>
}
