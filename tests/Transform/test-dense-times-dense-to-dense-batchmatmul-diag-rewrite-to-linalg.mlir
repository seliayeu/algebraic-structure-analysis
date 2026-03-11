// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite | FileCheck %s

// CHECK: #map = affine_map<(d0, d1) -> (d0, d1, d1)>

// CHECK-LABEL: func.func @main
// CHECK:         %[[EMPTY:.*]] = tensor.empty()
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [#map, #map, #map]
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]
// CHECK-SAME:      ins(%{{.*}}, %{{.*}} : tensor<2x3x3xf32>, tensor<2x3x3xf32>)
// CHECK-SAME:      outs(%[[EMPTY]] : tensor<2x3x3xf32>)
// CHECK-SAME:      {metadata = {lowerBw = 0 : i64, propertyDims = [1, 2], upperBw = 0 : i64}}
// CHECK:           ^bb0(%[[A:.*]]: f32, %[[B:.*]]: f32, %{{.*}}: f32):
// CHECK:             %[[MUL:.*]] = arith.mulf %[[A]], %[[B]] : f32
// CHECK:             linalg.yield %[[MUL]] : f32
// CHECK-NOT:     linalg.batch_matmul

module {
  func.func @main() -> tensor<2x3x3xf32> {
    %0 = tensor.empty() : tensor<2x3x3xf32>
    %A = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [1, 2]}}
            dense<[[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]],
                   [[2.0,0.0,0.0],[0.0,4.0,0.0],[0.0,0.0,7.0]]]> : tensor<2x3x3xf32>
    %B = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [1, 2]}}
            dense<[[[2.0,0.0,0.0],[0.0,4.0,0.0],[0.0,0.0,7.0]],
                   [[2.0,0.0,0.0],[0.0,4.0,0.0],[0.0,0.0,7.0]]]> : tensor<2x3x3xf32>
    %C = linalg.batch_matmul
      ins(%A, %B: tensor<2x3x3xf32>, tensor<2x3x3xf32>)
      outs(%0 : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
    return %C : tensor<2x3x3xf32>
  }
}
