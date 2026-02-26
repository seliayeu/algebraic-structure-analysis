// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite | FileCheck %s

// CHECK: #map = affine_map<(d0) -> (d0, d0)>
// CHECK: #map1 = affine_map<(d0) -> (d0)>

// CHECK: func.func @main() -> tensor<4xf32>

// CHECK: tensor.empty() : tensor<4xf32>

// CHECK: linalg.generic
// CHECK-SAME: indexing_maps = [#map, #map, #map1]
// CHECK-SAME: iterator_types = ["parallel"]
// CHECK-SAME: metadata = {layout = "dia", lowerBw = 0 : i64
// CHECK-SAME: upperBw = 0 : i64
// CHECK-NEXT: ^bb0(%{{.*}}: f32, %{{.*}}: f32, %{{.*}}: f32):
// CHECK-NEXT:   arith.mulf
// CHECK-NEXT:   linalg.yield

// CHECK: return %{{.*}} : tensor<4xf32>

module {
  func.func @main() -> tensor<4x4xf32> {
    %0 = tensor.empty() : tensor<4x4xf32>
    %diag = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0,0.0,0.0,0.0],
               [0.0,2.0,0.0,0.0],
               [0.0,0.0,3.0,0.0],
               [0.0,0.0,0.0,4.0]]> : tensor<4x4xf32>
    %tridiag = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0,0.0,0.0,0.0],
               [0.0,2.0,0.0,0.0],
               [0.0,0.0,3.0,0.0],
               [0.0,0.0,0.0,4.0]]> : tensor<4x4xf32>
    %4 = linalg.matmul ins(%diag, %tridiag : tensor<4x4xf32>, tensor<4x4xf32>)
                       outs(%0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %4 : tensor<4x4xf32>
  }
}
