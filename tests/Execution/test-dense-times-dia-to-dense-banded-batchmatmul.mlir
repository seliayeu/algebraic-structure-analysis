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
  // Combination 1: dense (L2, U0) * dia (L2, U1) = dense (L3, U1)
  // Total Bandwidth = 3 + 1 = 4 >= 4
  // ---------------------------------------------------------------------------
  // CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 4, 4]
  // CHECK-NEXT:    {{\[\[\[}}1, 1, 0, 0],
  // CHECK-NEXT:      [2, 2, 1, 0],
  // CHECK-NEXT:      [3, 3, 2, 1],
  // CHECK-NEXT:      [2, 3, 3, 2]],
  // CHECK-NEXT:     {{\[\[}}1, 1, 0, 0],
  // CHECK-NEXT:      [2, 2, 1, 0],
  // CHECK-NEXT:      [3, 3, 2, 1],
  // CHECK-NEXT:      [2, 3, 3, 2]]]
  func.func @test_dense_l2u0_dia_l2u1() {
    %A = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<[
      [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]], 
      [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0]]
    ]> : tensor<2x4x4xf32> 
    %B = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [1, 2], dia = true}} dense<[
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]], 
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]]
    ]> : tensor<2x4x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x4x4xf32>

    %R = dia.batch_matmul ins(%A, %B : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

    %memref = memref.alloc() : memref<2x4x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x4x4xf32>, memref<2x4x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x4x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x4x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Combination 2: dense (L1, U1) * dia (L1, U2) = dense (L2, U3)
  // Total Bandwidth = 2 + 3 = 5 >= 4
  // ---------------------------------------------------------------------------
  // CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 4, 4]
  // CHECK-NEXT:    {{\[\[\[}}2, 2, 2, 1],
  // CHECK-NEXT:      [2, 3, 3, 2],
  // CHECK-NEXT:      [1, 2, 3, 3],
  // CHECK-NEXT:      [0, 1, 2, 2]],
  // CHECK-NEXT:     {{\[\[}}2, 2, 2, 1],
  // CHECK-NEXT:      [2, 3, 3, 2],
  // CHECK-NEXT:      [1, 2, 3, 3],
  // CHECK-NEXT:      [0, 1, 2, 2]]]
  func.func @test_dense_l1u1_dia_l1u2() {
    %A = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} dense<[
      [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]], 
      [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]
    ]> : tensor<2x4x4xf32> 
    %B = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [1, 2], dia = true}} dense<[
      [[0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], 
      [[0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    ]> : tensor<2x4x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x4x4xf32>

    %R = dia.batch_matmul ins(%A, %B : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

    %memref = memref.alloc() : memref<2x4x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x4x4xf32>, memref<2x4x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x4x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x4x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Combination 3: dense (L0, U2) * dia (L3, U0) = dense (L3, U2)
  // Total Bandwidth = 3 + 2 = 5 >= 4
  // ---------------------------------------------------------------------------
  // CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 4, 4]
  // CHECK-NEXT:    {{\[\[\[}}3, 2, 1, 0],
  // CHECK-NEXT:      [3, 3, 2, 1],
  // CHECK-NEXT:      [2, 2, 2, 1],
  // CHECK-NEXT:      [1, 1, 1, 1]],
  // CHECK-NEXT:     {{\[\[}}3, 2, 1, 0],
  // CHECK-NEXT:      [3, 3, 2, 1],
  // CHECK-NEXT:      [2, 2, 2, 1],
  // CHECK-NEXT:      [1, 1, 1, 1]]]
  func.func @test_dense_l0u2_dia_l3u0() {
    %A = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} dense<[
      [[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]], 
      [[1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
    ]> : tensor<2x4x4xf32> 
    %B = arith.constant {metadata = {lowerBw = 3 : i64, upperBw = 0 : i64, propertyDims = [1, 2], dia = true}} dense<[
      [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]], 
      [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    ]> : tensor<2x4x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x4x4xf32>

    %R = dia.batch_matmul ins(%A, %B : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

    %memref = memref.alloc() : memref<2x4x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x4x4xf32>, memref<2x4x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x4x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x4x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Combination 4: dense (L2, U2) * dia (L2, U2) = dense (L3, U3)
  // Total Bandwidth = 3 + 3 = 6 >= 4
  // ---------------------------------------------------------------------------
  // CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 4, 4]
  // CHECK-NEXT:    {{\[\[\[}}3, 3, 3, 2],
  // CHECK-NEXT:      [3, 4, 4, 3],
  // CHECK-NEXT:      [3, 4, 4, 3],
  // CHECK-NEXT:      [2, 3, 3, 3]],
  // CHECK-NEXT:     {{\[\[}}3, 3, 3, 2],
  // CHECK-NEXT:      [3, 4, 4, 3],
  // CHECK-NEXT:      [3, 4, 4, 3],
  // CHECK-NEXT:      [2, 3, 3, 3]]]
  func.func @test_dense_l2u2_dia_l2u2() {
    %A = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} dense<[
      [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]], 
      [[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]]
    ]> : tensor<2x4x4xf32> 
    %B = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 2 : i64, propertyDims = [1, 2], dia = true}} dense<[
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]], 
      [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]
    ]> : tensor<2x5x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x4x4xf32>

    %R = dia.batch_matmul ins(%A, %B : tensor<2x4x4xf32>, tensor<2x5x4xf32>) outs(%zeroes : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

    %memref = memref.alloc() : memref<2x4x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x4x4xf32>, memref<2x4x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x4x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x4x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Combination 5: dense (L3, U0) * dia (L0, U3) = dense (L3, U3)
  // Total Bandwidth = 3 + 3 = 6 >= 4
  // ---------------------------------------------------------------------------
  // CHECK: Unranked Memref {{.*}} rank = 3 offset = 0 sizes = [2, 4, 4]
  // CHECK-NEXT:    {{\[\[\[}}1, 1, 1, 1],
  // CHECK-NEXT:      [1, 2, 2, 2],
  // CHECK-NEXT:      [1, 2, 3, 3],
  // CHECK-NEXT:      [1, 2, 3, 4]],
  // CHECK-NEXT:     {{\[\[}}1, 1, 1, 1],
  // CHECK-NEXT:      [1, 2, 2, 2],
  // CHECK-NEXT:      [1, 2, 3, 3],
  // CHECK-NEXT:      [1, 2, 3, 4]]]
  func.func @test_dense_l3u0_dia_l0u3() {
    %A = arith.constant {metadata = {lowerBw = 3 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<[
      [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]], 
      [[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 1.0]]
    ]> : tensor<2x4x4xf32> 
    %B = arith.constant {metadata = {dia = true, lowerBw = 0 : i64, upperBw = 3 : i64, propertyDims = [1, 2]}} dense<[
      [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], 
      [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]
    ]> : tensor<2x4x4xf32> 
    %zeroes = arith.constant dense<0.0> : tensor<2x4x4xf32>

    %R = dia.batch_matmul ins(%A, %B : tensor<2x4x4xf32>, tensor<2x4x4xf32>) outs(%zeroes : tensor<2x4x4xf32>) -> tensor<2x4x4xf32>

    %memref = memref.alloc() : memref<2x4x4xf32>
    bufferization.materialize_in_destination %R in writable %memref : (tensor<2x4x4xf32>, memref<2x4x4xf32>) -> ()

    %cast = memref.cast %memref : memref<2x4x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x4x4xf32>
    return
  }

  // ---------------------------------------------------------------------------
  // Main execution sequence
  // ---------------------------------------------------------------------------
  func.func @main() {
    call @test_dense_l2u0_dia_l2u1() : () -> ()
    call @test_dense_l1u1_dia_l1u2() : () -> ()
    call @test_dense_l0u2_dia_l3u0() : () -> ()
    call @test_dense_l2u2_dia_l2u2() : () -> ()
    call @test_dense_l3u0_dia_l0u3() : () -> ()
    return
  }
}
