// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite | FileCheck %s

#map  = affine_map<(d0) -> (0, d0)>
#map1 = affine_map<(d0) -> (d0, d0)>

// CHECK-DAG: #map = affine_map<(d0) -> (0, d0)>
// CHECK-DAG: #map1 = affine_map<(d0) -> (d0, d0)>

// CHECK-LABEL: func.func @main
// CHECK-SAME: -> tensor<1x3xf32>
module {
  func.func @main() -> tensor<1x3xf32> {
    %dia1 = arith.constant {metadata = {dia = true, upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 3.0]]> : tensor<1x3xf32>
    %dia2 = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 0.0],
               [0.0, 2.0, 0.0],
               [0.0, 0.0, 3.0]]> : tensor<3x3xf32>
    %0 = tensor.empty() : tensor<1x3xf32>
    %1 = dia.matmul ins(%dia1, %dia2 : tensor<1x3xf32>, tensor<3x3xf32>)
                    outs(%0 : tensor<1x3xf32>) -> tensor<1x3xf32>
    // CHECK: linalg.generic
    // CHECK-SAME: indexing_maps = [#map, #map1, #map]
    // CHECK-SAME: iterator_types = ["parallel"]
    // CHECK-SAME: metadata = {dia = true, lowerBw = 0 : i64, propertyDims = [0, 1], upperBw = 0 : i64}
    // CHECK: arith.mulf
    // CHECK: linalg.yield
    // CHECK-NOT: dia.matmul
    return %1 : tensor<1x3xf32>
  }
}
