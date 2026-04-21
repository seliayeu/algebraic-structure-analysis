// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite | FileCheck %s

// CHECK-LABEL: func.func @main() -> tensor<3x3xf32>

// Both DIA storage tensors must survive into the lowered output.
// CHECK-DAG: %[[DIAA:.*]] = arith.constant {{.*}} dense<{{.*}}> : tensor<2x3xf32>
// CHECK-DAG: %[[DIAB:.*]] = arith.constant {{.*}} dense<{{.*}}> : tensor<2x3xf32>

// Output is zero-initialised before the loop nest.
// CHECK: linalg.fill ins(%{{.*}} : f32) outs(%{{.*}} : tensor<3x3xf32>)

// Loop structure (from outermost to innermost):
//   %[[ROW]]  — iterates over output rows
//   %[[OCOL]] — iterates over output columns  (written to C[ROW, OCOL])
//   %[[K]]    — reduction index (contracts A and B diagonals)
//
// CHECK: scf.for %[[ROW:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK:   scf.for %[[OCOL:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK:     scf.for %[[K:.*]] = %{{.*}} to %{{.*}} step %{{.*}}

// DIA coordinate arithmetic inside the innermost body.
// All three loop indices are cast to i64 for offset computation.
// CHECK:       arith.index_cast %[[ROW]]  : index to i64
// CHECK:       arith.index_cast %[[K]]    : index to i64
// CHECK:       arith.index_cast %[[OCOL]] : index to i64
// CHECK:       arith.subi
// CHECK:       arith.addi

// Reads from both DIA storage tensors.
// CHECK:       tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<2x3xf32>
// CHECK:       tensor.extract %{{.*}}[%{{.*}}, %{{.*}}] : tensor<2x3xf32>

// Multiply-accumulate into C[row, outputCol].
// The insert index is [ROW, OCOL] — the inner K loop is the reduction and
// does NOT appear in the output index.
// CHECK:       arith.mulf %{{.*}}, %{{.*}} : f32
// CHECK:       arith.addf %{{.*}}, %{{.*}} : f32
// CHECK:       tensor.insert %{{.*}} into %{{.*}}[%[[ROW]], %[[OCOL]]]

// The original op must have been fully rewritten.
// CHECK-NOT: dia.matmul

module {
  func.func @main() -> tensor<3x3xf32> {
    %0 = tensor.empty() : tensor<3x3xf32>
    %diaA = arith.constant
        {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 4.0, 8.0],
               [1.0, 5.0, 9.0]]> : tensor<2x3xf32>
    %diaB = arith.constant
        {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 5.0, 9.0],
               [2.0, 6.0, 0.0]]> : tensor<2x3xf32>
    %1 = dia.matmul ins(%diaA, %diaB : tensor<2x3xf32>, tensor<2x3xf32>)
                    outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
    return %1 : tensor<3x3xf32>
  }
}
