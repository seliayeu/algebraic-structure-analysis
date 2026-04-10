// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite \
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
  // Combination 1: dia (L2, U0) * dense (L2, U0) = dia (batch x 4 x N)
  // (L=4 clamped to L=3 for 4x4 matrix)
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
  func.func @test_dia_l2u0_dense_l2u0() {
    %A = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<[
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], 
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    ]> : tensor<2x3x4xf32> 
    %B = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x4x4xf32>

    %R = dia.batch_matmul ins(%A, %B : tensor<2x3x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

    %memref = memref.alloc() : memref<2x4x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x4x4xf32>, memref<2x4x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x4x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x4x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Combination 2: dia (L2, U0) * dense (L0, U2) = dia (batch x 5 x N)
  // (L=2, U=2 -> 5 diagonals, no clamping needed)
  // ---------------------------------------------------------------------------
  // CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 5, 4]
  // CHECK-NEXT:    {{\[\[\[}}0, 0, 1, 1],
  // CHECK-NEXT:      [0, 1, 2, 2],
  // CHECK-NEXT:      [1, 2, 3, 3],
  // CHECK-NEXT:      [1, 2, 2, 0],
  // CHECK-NEXT:      [1, 1, 0, 0]],
  // CHECK-NEXT:     {{\[\[}}0, 0, 1, 1],
  // CHECK-NEXT:      [0, 1, 2, 2],
  // CHECK-NEXT:      [1, 2, 3, 3],
  // CHECK-NEXT:      [1, 2, 2, 0],
  // CHECK-NEXT:      [1, 1, 0, 0]]]
  func.func @test_dia_l2u0_dense_l0u2() {
    %A = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<[
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], 
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    ]> : tensor<2x3x4xf32> 
    %B = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x5x4xf32>

    %R = dia.batch_matmul ins(%A, %B : tensor<2x3x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x5x4xf32>) -> tensor<2x5x4xf32>

    %memref = memref.alloc() : memref<2x5x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x5x4xf32>, memref<2x5x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x5x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x5x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Combination 3: dia (L0, U2) * dense (L2, U0) = dia (batch x 5 x N)
  // (L=2, U=2 -> 5 diagonals, no clamping needed)
  // ---------------------------------------------------------------------------
  // CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 5, 4]
  // CHECK-NEXT:    {{\[\[\[}}0, 0, 1, 1],
  // CHECK-NEXT:      [0, 2, 2, 1],
  // CHECK-NEXT:      [3, 3, 2, 1],
  // CHECK-NEXT:      [2, 2, 1, 0],
  // CHECK-NEXT:      [1, 1, 0, 0]],
  // CHECK-NEXT:     {{\[\[}}0, 0, 1, 1],
  // CHECK-NEXT:      [0, 2, 2, 1],
  // CHECK-NEXT:      [3, 3, 2, 1],
  // CHECK-NEXT:      [2, 2, 1, 0],
  // CHECK-NEXT:      [1, 1, 0, 0]]]
  func.func @test_dia_l0u2_dense_l2u0() {
    %A = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} dense<[
      [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], 
      [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    ]> : tensor<2x3x4xf32> 
    %B = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x5x4xf32>

    %R = dia.batch_matmul ins(%A, %B : tensor<2x3x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x5x4xf32>) -> tensor<2x5x4xf32>

    %memref = memref.alloc() : memref<2x5x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x5x4xf32>, memref<2x5x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x5x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x5x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Combination 4: dia (L1, U1) * dense (L1, U1) = dia (batch x 5 x N)
  // (L=2, U=2 -> 5 diagonals, no clamping needed)
  // ---------------------------------------------------------------------------
  // CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 5, 4]
  // CHECK-NEXT:    {{\[\[\[}}0, 0, 1, 1],
  // CHECK-NEXT:      [0, 2, 2, 2],
  // CHECK-NEXT:      [2, 3, 3, 2],
  // CHECK-NEXT:      [2, 2, 2, 0],
  // CHECK-NEXT:      [1, 1, 0, 0]],
  // CHECK-NEXT:     {{\[\[}}0, 0, 1, 1],
  // CHECK-NEXT:      [0, 2, 2, 2],
  // CHECK-NEXT:      [2, 3, 3, 2],
  // CHECK-NEXT:      [2, 2, 2, 0],
  // CHECK-NEXT:      [1, 1, 0, 0]]]
  func.func @test_dia_l1u1_dense_l1u1() {
    %A = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} dense<[
      [[0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]], 
      [[0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]]
    ]> : tensor<2x3x4xf32> 
    %B = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x5x4xf32>

    %R = dia.batch_matmul ins(%A, %B : tensor<2x3x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x5x4xf32>) -> tensor<2x5x4xf32>

    %memref = memref.alloc() : memref<2x5x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x5x4xf32>, memref<2x5x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x5x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x5x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Combination 5: dia (L0, U0) * dense (L0, U0) = dia (batch x 1 x N)
  // ---------------------------------------------------------------------------
  // CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 1, 4]
  // CHECK-NEXT:    {{\[\[\[}}1, 1, 1, 1]],
  // CHECK-NEXT:     {{\[\[}}1, 1, 1, 1]]]
  func.func @test_dia_l0u0_dense_l0u0() {
    %A = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<[
      [[1.0, 1.0, 1.0, 1.0]], 
      [[1.0, 1.0, 1.0, 1.0]]
    ]> : tensor<2x1x4xf32> 
    %B = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<2x4x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x1x4xf32>

    %R = dia.batch_matmul ins(%A, %B : tensor<2x1x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x1x4xf32>) -> tensor<2x1x4xf32>

    %memref = memref.alloc() : memref<2x1x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x1x4xf32>, memref<2x1x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x1x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x1x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Main execution sequence
  // ---------------------------------------------------------------------------
  func.func @main() {
    call @test_dia_l2u0_dense_l2u0() : () -> ()
    call @test_dia_l2u0_dense_l0u2() : () -> ()
    call @test_dia_l0u2_dense_l2u0() : () -> ()
    call @test_dia_l1u1_dense_l1u1() : () -> ()
    call @test_dia_l0u0_dense_l0u0() : () -> ()
    return
  }
}
