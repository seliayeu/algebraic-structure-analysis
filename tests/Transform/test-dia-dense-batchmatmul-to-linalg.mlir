// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite | FileCheck %s

module {
  // CHECK-LABEL: func.func @main() -> tensor<2x4x4xf32> {
  func.func @main() -> tensor<2x4x4xf32> {
    // CHECK-DAG: %[[A:.*]] = arith.constant {metadata = {lowerBw = 2 : i64, propertyDims = [1, 2], upperBw = 2 : i64}} dense{{.*}} : tensor<2x4x4xf32>
    %A = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} dense<[
      [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]],
      [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]]
    ]> : tensor<2x4x4xf32> 

    %B = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} dense<[
      [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]],
      [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]]
    ]> : tensor<2x4x4xf32> 

    // CHECK-DAG: %[[ZEROES:.*]] = arith.constant {{.*}} dense<0.000000e+00> : tensor<2x4x4xf32>
    %zeroes = arith.constant dense<0.0> : tensor<2x4x4xf32>

    // CHECK-DAG: %[[C0:.*]] = arith.constant 0 : index
    // CHECK-DAG: %[[C1:.*]] = arith.constant 1 : index
    // CHECK-DAG: %[[C2:.*]] = arith.constant 2 : index
    // CHECK-DAG: %[[C4:.*]] = arith.constant 4 : index

    // Verify outer batch loop
    // CHECK: scf.for %[[BATCH:.*]] = %[[C0]] to %[[C2]] step %[[C1]] iter_args(%[[ITER1:.*]] = %[[ZEROES]]) -> (tensor<2x4x4xf32>) {
    
    // Verify row loop (M)
    // CHECK:   scf.for %[[M:.*]] = %[[C0]] to %[[C4]] step %[[C1]] iter_args(%[[ITER2:.*]] = %[[ITER1]]) -> (tensor<2x4x4xf32>) {
    
    // Verify boundaries calculations for N
    // CHECK:     %[[MIN_N:.*]] = arith.maxsi %[[C0]], {{.*}} : index
    // CHECK:     %[[MAX_N:.*]] = arith.minsi {{.*}}, {{.*}} : index

    // Verify column loop (N)
    // CHECK:     scf.for %[[N:.*]] = %[[MIN_N]] to %[[MAX_N]] step %[[C1]] iter_args(%[[ITER3:.*]] = %[[ITER2]]) -> (tensor<2x4x4xf32>) {
    
    // Verify boundaries calculations for K
    // CHECK:       %[[MIN_K:.*]] = arith.maxsi %[[C0]], {{.*}} : index
    // CHECK:       %[[MAX_K:.*]] = arith.minsi %[[C4]], {{.*}} : index
    
    // Verify inner reduction loop (K)
    // CHECK:       scf.for %[[K:.*]] = %[[MIN_K]] to %[[MAX_K]] step %[[C1]] iter_args(%[[ITER4:.*]] = %[[ITER3]]) -> (tensor<2x4x4xf32>) {
    
    // Verify element extraction, math, and insertion
    // CHECK:         %[[VAL_C:.*]] = tensor.extract %[[ITER4]][%[[BATCH]], %[[M]], %[[N]]] : tensor<2x4x4xf32>
    // CHECK:         %[[VAL_A:.*]] = tensor.extract %[[A]][%[[BATCH]], %[[M]], %[[K]]] : tensor<2x4x4xf32>
    // CHECK:         %[[VAL_B:.*]] = tensor.extract %[[A]][%[[BATCH]], %[[K]], %[[N]]] : tensor<2x4x4xf32>
    // CHECK:         %[[MUL:.*]] = arith.mulf %[[VAL_A]], %[[VAL_B]] : f32
    // CHECK:         %[[ADD:.*]] = arith.addf %[[VAL_C]], %[[MUL]] : f32
    // CHECK:         %[[INSERT:.*]] = tensor.insert %[[ADD]] into %[[ITER4]][%[[BATCH]], %[[M]], %[[N]]] : tensor<2x4x4xf32>
    
    // CHECK:         scf.yield %[[INSERT]] : tensor<2x4x4xf32>
    // CHECK:       }
    // CHECK:       scf.yield {{.*}} : tensor<2x4x4xf32>
    // CHECK:     }
    // CHECK:     scf.yield {{.*}} : tensor<2x4x4xf32>
    // CHECK:   }
    // CHECK:   scf.yield {{.*}} : tensor<2x4x4xf32>
    // CHECK: }
    
    %R = dia.batch_matmul ins(%A, %B : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>
    
    // CHECK: return {{.*}} : tensor<2x4x4xf32>
    return %R: tensor<2x4x4xf32>
  }
}
