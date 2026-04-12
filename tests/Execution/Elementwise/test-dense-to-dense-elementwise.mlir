// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format | \
// RUN:  FileCheck %s --check-prefix=EXEC

func.func private @printMemrefF32(memref<*xf32>)

// ===----------------------------------------------------------------------===//
// Affine map expected after 3D diagonal lowering:
//   (d0, d1) -> (d0, d1, d1)   i.e. batch + diagonal access
// ===----------------------------------------------------------------------===//
// CHECK-DAG: #[[DIAG:.*]] = affine_map<(d0, d1) -> (d0, d1, d1)>

// ===----------------------------------------------------------------------===//
// 1. 3D diagonal add  ->  linalg.generic with batch+diag iterators
// ===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_3d_diagonal_add
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [#[[DIAG]], #[[DIAG]], #[[DIAG]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]
// CHECK:             arith.addf
// CHECK-NOT:     scf.for
func.func @test_3d_diagonal_add() -> tensor<2x8x8xf32> {
  %A = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      dense<1.0> : tensor<2x8x8xf32>
  %B = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      dense<2.0> : tensor<2x8x8xf32>
  %zero = arith.constant 0.0 : f32
  %C_init = tensor.empty() : tensor<2x8x8xf32>
  %C = linalg.fill ins(%zero : f32) outs(%C_init : tensor<2x8x8xf32>) -> tensor<2x8x8xf32>
  %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
      {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      ins(%A, %B : tensor<2x8x8xf32>, tensor<2x8x8xf32>)
      outs(%C : tensor<2x8x8xf32>) -> tensor<2x8x8xf32>
  return %result : tensor<2x8x8xf32>
}

// ===----------------------------------------------------------------------===//
// 2. 3D diagonal sub  ->  linalg.generic, arith.subf inside
// ===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_3d_diagonal_sub
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [#[[DIAG]], #[[DIAG]], #[[DIAG]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]
// CHECK:             arith.subf
// CHECK-NOT:     scf.for
func.func @test_3d_diagonal_sub() -> tensor<2x8x8xf32> {
  %A = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      dense<3.0> : tensor<2x8x8xf32>
  %B = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      dense<1.0> : tensor<2x8x8xf32>
  %zero = arith.constant 0.0 : f32
  %C_init = tensor.empty() : tensor<2x8x8xf32>
  %C = linalg.fill ins(%zero : f32) outs(%C_init : tensor<2x8x8xf32>) -> tensor<2x8x8xf32>
  %result = linalg.elementwise kind=#linalg.elementwise_kind<sub>
      {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      ins(%A, %B : tensor<2x8x8xf32>, tensor<2x8x8xf32>)
      outs(%C : tensor<2x8x8xf32>) -> tensor<2x8x8xf32>
  return %result : tensor<2x8x8xf32>
}

// ===----------------------------------------------------------------------===//
// 3. 3D diagonal mul  ->  linalg.generic, arith.mulf inside
// ===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_3d_diagonal_mul
// CHECK:         linalg.generic
// CHECK-SAME:      indexing_maps = [#[[DIAG]], #[[DIAG]], #[[DIAG]]]
// CHECK-SAME:      iterator_types = ["parallel", "parallel"]
// CHECK:             arith.mulf
// CHECK-NOT:     scf.for
func.func @test_3d_diagonal_mul() -> tensor<2x8x8xf32> {
  %A = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      dense<2.0> : tensor<2x8x8xf32>
  %B = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      dense<3.0> : tensor<2x8x8xf32>
  %zero = arith.constant 0.0 : f32
  %C_init = tensor.empty() : tensor<2x8x8xf32>
  %C = linalg.fill ins(%zero : f32) outs(%C_init : tensor<2x8x8xf32>) -> tensor<2x8x8xf32>
  %result = linalg.elementwise kind=#linalg.elementwise_kind<mul>
      {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      ins(%A, %B : tensor<2x8x8xf32>, tensor<2x8x8xf32>)
      outs(%C : tensor<2x8x8xf32>) -> tensor<2x8x8xf32>
  return %result : tensor<2x8x8xf32>
}

// ===----------------------------------------------------------------------===//
// 4. 3D banded (tridiagonal) add  ->  batch scf.for wrapping 2 inner loops
// ===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_3d_banded_add
// CHECK:         scf.for %[[B:[a-zA-Z0-9_]+]] =
// CHECK:           scf.for %[[I:[a-zA-Z0-9_]+]] =
// CHECK:             scf.for %[[J:[a-zA-Z0-9_]+]] =
// CHECK:               tensor.extract %{{.*}}[%[[B]], %[[I]], %[[J]]]
// CHECK:               tensor.extract %{{.*}}[%[[B]], %[[I]], %[[J]]]
// CHECK:               arith.addf
// CHECK:               tensor.insert %{{.*}} into %{{.*}}[%[[B]], %[[I]], %[[J]]]
// CHECK-NOT:         linalg.generic
func.func @test_3d_banded_add() -> tensor<2x8x8xf32> {
  %A = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}}
      dense<1.0> : tensor<2x8x8xf32>
  %B = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}}
      dense<2.0> : tensor<2x8x8xf32>
  %zero = arith.constant 0.0 : f32
  %C_init = tensor.empty() : tensor<2x8x8xf32>
  %C = linalg.fill ins(%zero : f32) outs(%C_init : tensor<2x8x8xf32>) -> tensor<2x8x8xf32>
  %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
      {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}}
      ins(%A, %B : tensor<2x8x8xf32>, tensor<2x8x8xf32>)
      outs(%C : tensor<2x8x8xf32>) -> tensor<2x8x8xf32>
  return %result : tensor<2x8x8xf32>
}

// ===----------------------------------------------------------------------===//
// 5. 3D banded sub  ->  same structure, arith.subf inside
// ===----------------------------------------------------------------------===//
// CHECK-LABEL: func.func @test_3d_banded_sub
// CHECK:         scf.for
// CHECK:           scf.for
// CHECK:             scf.for
// CHECK:               arith.subf
// CHECK-NOT:         linalg.generic
func.func @test_3d_banded_sub() -> tensor<2x8x8xf32> {
  %A = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}}
      dense<3.0> : tensor<2x8x8xf32>
  %B = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}}
      dense<1.0> : tensor<2x8x8xf32>
  %zero = arith.constant 0.0 : f32
  %C_init = tensor.empty() : tensor<2x8x8xf32>
  %C = linalg.fill ins(%zero : f32) outs(%C_init : tensor<2x8x8xf32>) -> tensor<2x8x8xf32>
  %result = linalg.elementwise kind=#linalg.elementwise_kind<sub>
      {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}}
      ins(%A, %B : tensor<2x8x8xf32>, tensor<2x8x8xf32>)
      outs(%C : tensor<2x8x8xf32>) -> tensor<2x8x8xf32>
  return %result : tensor<2x8x8xf32>
}

// ===----------------------------------------------------------------------===//
// Execution tests (EXEC prefix)
// ===----------------------------------------------------------------------===//

// -----------------------------------------------------------------------
// EXEC 1: diagonal add
// -----------------------------------------------------------------------
// EXEC:      Unranked Memref {{.*}} data =
// EXEC-NEXT: {{.*\[\[\[}}10{{.*}}0{{.*}}0],
// EXEC-NEXT: {{.*\[}}0{{.*}}10{{.*}}0],
// EXEC-NEXT: {{.*\[}}0{{.*}}0{{.*}}10]],
// EXEC-NEXT: {{.*\[\[}}10{{.*}}0{{.*}}0],
// EXEC-NEXT: {{.*\[}}0{{.*}}10{{.*}}0],
// EXEC-NEXT: {{.*\[}}0{{.*}}0{{.*}}10]]]
func.func @exec_diagonal_add() {
  %A = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      dense<[[[1.0, 0.0, 0.0],
               [0.0, 2.0, 0.0],
               [0.0, 0.0, 3.0]],
              [[4.0, 0.0, 0.0],
               [0.0, 5.0, 0.0],
               [0.0, 0.0, 6.0]]]> : tensor<2x3x3xf32>
  %B = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      dense<[[[9.0, 0.0, 0.0],
               [0.0, 8.0, 0.0],
               [0.0, 0.0, 7.0]],
              [[6.0, 0.0, 0.0],
               [0.0, 5.0, 0.0],
               [0.0, 0.0, 4.0]]]> : tensor<2x3x3xf32>
  %zero = arith.constant 0.0 : f32
  %C_init = tensor.empty() : tensor<2x3x3xf32>
  %C = linalg.fill ins(%zero : f32) outs(%C_init : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
  %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
      {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      ins(%A, %B : tensor<2x3x3xf32>, tensor<2x3x3xf32>)
      outs(%C : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>

  %memref = memref.alloc() : memref<2x3x3xf32>
  bufferization.materialize_in_destination %result in %memref {writable} : (tensor<2x3x3xf32>, memref<2x3x3xf32>) -> ()
  %cast = memref.cast %memref : memref<2x3x3xf32> to memref<*xf32>

  call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
  memref.dealloc %memref : memref<2x3x3xf32>
  return
}

// -----------------------------------------------------------------------
// EXEC 2: diagonal mul
// -----------------------------------------------------------------------
// EXEC:      Unranked Memref {{.*}} data =
// EXEC-NEXT: {{.*\[\[\[}}10{{.*}}0{{.*}}0],
// EXEC-NEXT: {{.*\[}}0{{.*}}15{{.*}}0],
// EXEC-NEXT: {{.*\[}}0{{.*}}0{{.*}}20]],
// EXEC-NEXT: {{.*\[\[}}4{{.*}}0{{.*}}0],
// EXEC-NEXT: {{.*\[}}0{{.*}}8{{.*}}0],
// EXEC-NEXT: {{.*\[}}0{{.*}}0{{.*}}12]]]
func.func @exec_diagonal_mul() {
  %A = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      dense<[[[2.0, 0.0, 0.0],
               [0.0, 3.0, 0.0],
               [0.0, 0.0, 4.0]],
              [[1.0, 0.0, 0.0],
               [0.0, 2.0, 0.0],
               [0.0, 0.0, 3.0]]]> : tensor<2x3x3xf32>
  %B = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      dense<[[[5.0, 0.0, 0.0],
               [0.0, 5.0, 0.0],
               [0.0, 0.0, 5.0]],
              [[4.0, 0.0, 0.0],
               [0.0, 4.0, 0.0],
               [0.0, 0.0, 4.0]]]> : tensor<2x3x3xf32>
  %zero = arith.constant 0.0 : f32
  %C_init = tensor.empty() : tensor<2x3x3xf32>
  %C = linalg.fill ins(%zero : f32) outs(%C_init : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
  %result = linalg.elementwise kind=#linalg.elementwise_kind<mul>
      {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      ins(%A, %B : tensor<2x3x3xf32>, tensor<2x3x3xf32>)
      outs(%C : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
  %memref = memref.alloc() : memref<2x3x3xf32>
  bufferization.materialize_in_destination %result in %memref {writable} : (tensor<2x3x3xf32>, memref<2x3x3xf32>) -> ()
  %cast = memref.cast %memref : memref<2x3x3xf32> to memref<*xf32>

  call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
  memref.dealloc %memref : memref<2x3x3xf32>
  return
}

// -----------------------------------------------------------------------
// EXEC 3: banded (tridiagonal, lower=1 upper=1) add
// -----------------------------------------------------------------------
// EXEC:      Unranked Memref {{.*}} data =
// EXEC-NEXT: {{.*\[\[\[}}3{{.*}}3{{.*}}0],
// EXEC-NEXT: {{.*\[}}3{{.*}}3{{.*}}3],
// EXEC-NEXT: {{.*\[}}0{{.*}}3{{.*}}3]],
// EXEC-NEXT: {{.*\[\[}}3{{.*}}3{{.*}}0],
// EXEC-NEXT: {{.*\[}}3{{.*}}3{{.*}}3],
// EXEC-NEXT: {{.*\[}}0{{.*}}3{{.*}}3]]]
func.func @exec_banded_add() {
  %A = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}}
      dense<[[[1.0, 1.0, 0.0],
               [1.0, 1.0, 1.0],
               [0.0, 1.0, 1.0]],
              [[1.0, 1.0, 0.0],
               [1.0, 1.0, 1.0],
               [0.0, 1.0, 1.0]]]> : tensor<2x3x3xf32>
  %B = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}}
      dense<[[[2.0, 2.0, 0.0],
               [2.0, 2.0, 2.0],
               [0.0, 2.0, 2.0]],
              [[2.0, 2.0, 0.0],
               [2.0, 2.0, 2.0],
               [0.0, 2.0, 2.0]]]> : tensor<2x3x3xf32>
  %zero = arith.constant 0.0 : f32
  %C_init = tensor.empty() : tensor<2x3x3xf32>
  %C = linalg.fill ins(%zero : f32) outs(%C_init : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
  %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
      {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}}
      ins(%A, %B : tensor<2x3x3xf32>, tensor<2x3x3xf32>)
      outs(%C : tensor<2x3x3xf32>) -> tensor<2x3x3xf32>
  %memref = memref.alloc() : memref<2x3x3xf32>
  bufferization.materialize_in_destination %result in %memref {writable} : (tensor<2x3x3xf32>, memref<2x3x3xf32>) -> ()
  %cast = memref.cast %memref : memref<2x3x3xf32> to memref<*xf32>

  call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
  memref.dealloc %memref : memref<2x3x3xf32>
  return
}

// -----------------------------------------------------------------------
// EXEC 4: asymmetric band (lower=2, upper=0) add
// -----------------------------------------------------------------------
// EXEC:      Unranked Memref {{.*}} data =
// EXEC-NEXT: {{.*\[\[\[}}10{{.*}}0{{.*}}0],
// EXEC-NEXT: {{.*\[}}10{{.*}}10{{.*}}0],
// EXEC-NEXT: {{.*\[}}10{{.*}}10{{.*}}10]]]
func.func @exec_banded_lower_only() {
  %A = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      dense<[[[1.0, 0.0, 0.0],
               [2.0, 3.0, 0.0],
               [4.0, 5.0, 6.0]]]> : tensor<1x3x3xf32>
  %B = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      dense<[[[9.0, 0.0, 0.0],
               [8.0, 7.0, 0.0],
               [6.0, 5.0, 4.0]]]> : tensor<1x3x3xf32>
  %zero = arith.constant 0.0 : f32
  %C_init = tensor.empty() : tensor<1x3x3xf32>
  %C = linalg.fill ins(%zero : f32) outs(%C_init : tensor<1x3x3xf32>) -> tensor<1x3x3xf32>
  %result = linalg.elementwise kind=#linalg.elementwise_kind<add>
      {metadata = {lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}}
      ins(%A, %B : tensor<1x3x3xf32>, tensor<1x3x3xf32>)
      outs(%C : tensor<1x3x3xf32>) -> tensor<1x3x3xf32>


  %memref = memref.alloc() : memref<1x3x3xf32>
  bufferization.materialize_in_destination %result in %memref {writable} : (tensor<1x3x3xf32>, memref<1x3x3xf32>) -> ()
  %cast = memref.cast %memref : memref<1x3x3xf32> to memref<*xf32>

  call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
  memref.dealloc %memref : memref<1x3x3xf32>
  return
}

// ===----------------------------------------------------------------------===//
// main: entry point for mlir-cpu-runner
// ===----------------------------------------------------------------------===//
func.func @main() {
  func.call @exec_diagonal_add()      : () -> ()
  func.call @exec_diagonal_mul()      : () -> ()
  func.call @exec_banded_add()        : () -> ()
  func.call @exec_banded_lower_only() : () -> ()
  return
}
