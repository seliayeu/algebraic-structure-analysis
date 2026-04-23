// RUN: %build/tools/alg-opt %s --banded-analysis --canonicalize | FileCheck %s

// CHECK-LABEL: func.func @diagonal_only
// CHECK: %cst = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, propertyDims = [0, 1], upperBw = 0 : i64}} dense<{{\[\[}}1.{{.*}}, 2.{{.*}}, 3.{{.*}}]]> : tensor<1x3xf32>
// CHECK-NOT: dia.from_dense
func.func @diagonal_only() -> tensor<1x3xf32> {
  %dense = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
      dense<[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]> : tensor<3x3xf32>
  %dia = dia.from_dense %dense : tensor<3x3xf32> -> tensor<1x3xf32>
  return %dia : tensor<1x3xf32>
}

// CHECK-LABEL: func.func @upper_banded
// CHECK: %cst = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, propertyDims = [0, 1], upperBw = 1 : i64}} dense<{{\[\[}}1.{{.*}}, 3.{{.*}}, 5.{{.*}}], [2.{{.*}}, 4.{{.*}}, 0.{{.*}}]]> : tensor<2x3xf32>
// CHECK-NOT: dia.from_dense
func.func @upper_banded() -> tensor<2x3xf32> {
  %dense = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
      dense<[[1.0, 2.0, 0.0], [0.0, 3.0, 4.0], [0.0, 0.0, 5.0]]> : tensor<3x3xf32>
  %dia = dia.from_dense %dense : tensor<3x3xf32> -> tensor<2x3xf32>
  return %dia : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @lower_banded
// CHECK: %cst = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, propertyDims = [0, 1], upperBw = 0 : i64}} dense<{{\[\[}}0.{{.*}}, 2.{{.*}}, 4.{{.*}}], [1.{{.*}}, 3.{{.*}}, 5.{{.*}}]]> : tensor<2x3xf32>
// CHECK-NOT: dia.from_dense
func.func @lower_banded() -> tensor<2x3xf32> {
  %dense = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
      dense<[[1.0, 0.0, 0.0], [2.0, 3.0, 0.0], [0.0, 4.0, 5.0]]> : tensor<3x3xf32>
  %dia = dia.from_dense %dense : tensor<3x3xf32> -> tensor<2x3xf32>
  return %dia : tensor<2x3xf32>
}

// CHECK-LABEL: func.func @full_banded
// CHECK: %cst = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, propertyDims = [0, 1], upperBw = 1 : i64}} dense<{{\[\[}}0.{{.*}}, 3.{{.*}}, 6.{{.*}}], [1.{{.*}}, 4.{{.*}}, 7.{{.*}}], [2.{{.*}}, 5.{{.*}}, 0.{{.*}}]]> : tensor<3x3xf32>
// CHECK-NOT: dia.from_dense
func.func @full_banded() -> tensor<3x3xf32> {
  %dense = arith.constant {metadata = {upperBw = 1 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
      dense<[[1.0, 2.0, 0.0], [3.0, 4.0, 5.0], [0.0, 6.0, 7.0]]> : tensor<3x3xf32>
  %dia = dia.from_dense %dense : tensor<3x3xf32> -> tensor<3x3xf32>
  return %dia : tensor<3x3xf32>
}
