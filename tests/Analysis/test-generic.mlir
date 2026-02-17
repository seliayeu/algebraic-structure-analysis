// RUN: %build/tools/alg-opt %s --banded-structure-debug | FileCheck %s

module {
  func.func @test_generic() {
    %out = tensor.empty() : tensor<2x2x2xf32>

    %lhs_split = arith.constant { 
      metadata = { 
        upperBw = 1 : i64, 
        lowerBw = 1 : i64, 
        propertyDims = [0, 3]
      } 
    } dense<2.0> : tensor<2x1x2x2xf32>

    %rhs_std = arith.constant { 
      metadata = { 
        upperBw = 0 : i64, 
        lowerBw = 0 : i64, 
        propertyDims = [0, 1] 
      } 
    } dense<2.0> : tensor<2x2x1xf32>


    // CHECK: linalg.generic
    // CHECK-SAME: lowerBw = 1
    // CHECK-SAME: propertyDims = [0, 1]
    // CHECK-SAME: upperBw = 1
    %res_split = linalg.generic {
      indexing_maps = [
        affine_map<(a, b, c, d, f, g) -> (a, b, c, d)>, 
        affine_map<(a, b, c, d, f, g) -> (d, f, g)>, 
        affine_map<(a, b, c, d, f, g) -> (a, f, c)>
      ],
      iterator_types = ["parallel", "reduction", "parallel", "reduction", "parallel", "reduction"]
    } ins(%lhs_split, %rhs_std : tensor<2x1x2x2xf32>, tensor<2x2x1xf32>) outs(%out : tensor<2x2x2xf32>) {
    ^bb0(%in_lhs: f32, %in_rhs: f32, %acc: f32):
      %0 = arith.mulf %in_lhs, %in_rhs : f32
      %1 = arith.addf %acc, %0 : f32
      linalg.yield %1 : f32
    } -> tensor<2x2x2xf32>

    // CHECK: linalg.generic
    // CHECK-SAME: lowerBw = 1
    // CHECK-SAME: propertyDims = [2, 1]
    // CHECK-SAME: upperBw = 1
    %res_permuted = linalg.generic {
      indexing_maps = [
        affine_map<(a, b, c, d, f, g) -> (a, b, c, d)>, 
        affine_map<(a, b, c, d, f, g) -> (d, f, g)>, 
        affine_map<(a, b, c, d, f, g) -> (c, f, a)>
      ],
      iterator_types = ["parallel", "reduction", "parallel", "reduction", "parallel", "reduction"]
    } ins(%lhs_split, %rhs_std : tensor<2x1x2x2xf32>, tensor<2x2x1xf32>) outs(%out : tensor<2x2x2xf32>) {
    ^bb0(%in_lhs: f32, %in_rhs: f32, %acc: f32):
      %0 = arith.mulf %in_lhs, %in_rhs : f32
      %1 = arith.addf %acc, %0 : f32
      linalg.yield %1 : f32
    } -> tensor<2x2x2xf32>

    return
  }
}
