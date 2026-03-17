// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite | FileCheck %s

// CHECK-LABEL: func.func @test_dia_square
func.func @test_dia_square() -> tensor<2x4xf32> {
    %diaA = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}} 
        dense<[[1.0, 1.0, 1.0, 0.0], [2.0, 2.0, 2.0, 2.0]]> : tensor<2x4xf32>

    %0 = tensor.empty() : tensor<2x4xf32>
    
    // --- SQUARE Lowering ---
    // CHECK: %[[SQ_RES:.*]] = linalg.elementwise
    // CHECK-SAME: kind=#linalg.elementwise_kind<square>
    // CHECK-SAME: ins(%{{.*}} : tensor<2x4xf32>) outs(%{{.*}} : tensor<2x4xf32>)
    
    %1 = dia.elementwise kind = <square> 
         ins(%diaA : tensor<2x4xf32>) 
         outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>

    // CHECK: return %[[SQ_RES]] : tensor<2x4xf32>
    return %1 : tensor<2x4xf32>
}
