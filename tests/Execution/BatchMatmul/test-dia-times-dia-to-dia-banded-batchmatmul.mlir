// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format > %t
// RUN: FileCheck %s < %t

module {
  func.func private @printMemrefF32(memref<*xf32>)

  // ---------------------------------------------------------------------------
  // Combination 1: dia (L2, U0) * dia (L2, U0) = dia (L3, U0)
  // (Produces a batch x 4 x N dia tensor: L3, L2, L1, D0)
  // ---------------------------------------------------------------------------
  // CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 4, 4]
  // CHECK-NEXT:    {{\[\[\[}}0, 0, 0, 2],
  // CHECK-NEXT:      [0, 0, 3, 3],
  // CHECK-NEXT:      [0, 2, 2, 2],
  // CHECK-NEXT:      [1, 1, 1, 1]],
  // CHECK-NEXT:     {{\[\[}}0, 0, 0, 2],
  // CHECK-NEXT:      [0, 0, 3, 3],
  // CHECK-NEXT:      [0, 2, 2, 2],
  // CHECK-NEXT:      [1, 1, 1, 1]]]
  func.func @test_dia_l2u0_dia_l2u0_to_dia() {
    %A = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<[
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], 
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    ]> : tensor<2x3x4xf32> 

    %B = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<[
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], 
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    ]> : tensor<2x3x4xf32> 

    %zeroes = arith.constant {metadata = {dia = true, lowerBw = 3 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<0.0> : tensor<2x4x4xf32>

    %R = dia.batch_matmul ins(%A, %B : tensor<2x3x4xf32>, tensor<2x3x4xf32>) outs(%zeroes : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

    %memref = memref.alloc() : memref<2x4x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x4x4xf32>, memref<2x4x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x4x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x4x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Combination 2: dia (L0, U2) * dia (L0, U2) = dia (L0, U3)
  // (Produces a batch x 4 x N dia tensor: D0, U1, U2, U3)
  // ---------------------------------------------------------------------------
  // CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 4, 4]
  // CHECK-NEXT:    {{\[\[\[}}1, 1, 1, 1],
  // CHECK-NEXT:      [2, 2, 2, 0],
  // CHECK-NEXT:      [3, 3, 0, 0],
  // CHECK-NEXT:      [2, 0, 0, 0]],
  // CHECK-NEXT:     {{\[\[}}1, 1, 1, 1],
  // CHECK-NEXT:      [2, 2, 2, 0],
  // CHECK-NEXT:      [3, 3, 0, 0],
  // CHECK-NEXT:      [2, 0, 0, 0]]]
  func.func @test_dia_l0u2_dia_l0u2_to_dia() {
    %A = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} dense<[
      [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], 
      [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    ]> : tensor<2x3x4xf32> 

    %B = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} dense<[
      [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], 
      [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    ]> : tensor<2x3x4xf32> 

    %zeroes = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 3 : i64, propertyDims = [1, 2]}} dense<0.0> : tensor<2x4x4xf32>

    %R = dia.batch_matmul ins(%A, %B : tensor<2x3x4xf32>, tensor<2x3x4xf32>) outs(%zeroes : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

    %memref = memref.alloc() : memref<2x4x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x4x4xf32>, memref<2x4x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x4x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x4x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Combination 3: dia (L2, U2) * dia (L2, U2) = dia (L3, U3)
  // (Produces a batch x 7 x N dia tensor: L3, L2, L1, D0, U1, U2, U3)
  // ---------------------------------------------------------------------------
  // CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 7, 4]
  // CHECK-NEXT:    {{\[\[\[}}0, 0, 0, 2],
  // CHECK-NEXT:      [0, 0, 3, 3],
  // CHECK-NEXT:      [0, 3, 4, 3],
  // CHECK-NEXT:      [3, 4, 4, 3],
  // CHECK-NEXT:      [3, 4, 3, 0],
  // CHECK-NEXT:      [3, 3, 0, 0],
  // CHECK-NEXT:      [2, 0, 0, 0]],
  // CHECK-NEXT:     {{\[\[}}0, 0, 0, 2],
  // CHECK-NEXT:      [0, 0, 3, 3],
  // CHECK-NEXT:      [0, 3, 4, 3],
  // CHECK-NEXT:      [3, 4, 4, 3],
  // CHECK-NEXT:      [3, 4, 3, 0],
  // CHECK-NEXT:      [3, 3, 0, 0],
  // CHECK-NEXT:      [2, 0, 0, 0]]]
  func.func @test_dia_l2u2_dia_l2u2_to_dia() {
    %A = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} dense<[
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], 
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    ]> : tensor<2x5x4xf32> 

    %B = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} dense<[
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], 
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    ]> : tensor<2x5x4xf32> 

    %zeroes = arith.constant {metadata = {dia = true, lowerBw = 3 : i64, upperBw = 3 : i64, propertyDims = [1, 2]}} dense<0.0> : tensor<2x7x4xf32>

    %R = dia.batch_matmul ins(%A, %B : tensor<2x5x4xf32>, tensor<2x5x4xf32>) outs(%zeroes : tensor<2x7x4xf32>) -> tensor<2x7x4xf32>

    %memref = memref.alloc() : memref<2x7x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x7x4xf32>, memref<2x7x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x7x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x7x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Main execution sequence
  // ---------------------------------------------------------------------------
  func.func @main() {
    call @test_dia_l2u0_dia_l2u0_to_dia() : () -> ()
    call @test_dia_l0u2_dia_l0u2_to_dia() : () -> ()
    call @test_dia_l2u2_dia_l2u2_to_dia() : () -> ()
    return
  }
}
