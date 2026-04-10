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
  
  func.func @test_add() {
    %0 = tensor.empty() : tensor<5x5xf32>
    
    %denseA = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}} 
        dense<[[ 1.0,  2.0,  3.0,  0.0,  0.0],
               [ 4.0,  5.0,  6.0,  7.0,  0.0],
               [ 0.0,  8.0,  9.0, 10.0, 11.0],
               [ 0.0,  0.0, 12.0, 13.0, 14.0],
               [ 0.0,  0.0,  0.0, 15.0, 16.0]]> : tensor<5x5xf32>

    %diaB = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[ 0.0,  0.0,  6.0, 10.0, 14.0], // L2: 3 elements (rows 2, 3, 4)
               [ 0.0,  3.0,  7.0, 11.0, 15.0], // L1: 4 elements (rows 1, 2, 3, 4)
               [ 1.0,  4.0,  8.0, 12.0, 16.0], // M0: 5 elements (rows 0 to 4)
               [ 2.0,  5.0,  9.0, 13.0,  0.0]]> : tensor<4x5xf32> // U1: 4 elements (rows 0 to 3)

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [5, 5]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}2, 4, 3, 0, 0],
    // CHECK-NEXT:  [7, 9, 11, 7, 0],
    // CHECK-NEXT:  [6, 15, 17, 19, 11],
    // CHECK-NEXT:  [0, 10, 23, 25, 27],
    // CHECK-NEXT:  [0, 0, 14, 30, 32]]
    %2 = dia.elementwise kind = <add> ins(%denseA, %diaB : tensor<5x5xf32>, tensor<4x5xf32>) 
                                      outs(%0 : tensor<5x5xf32>) -> tensor<5x5xf32>
                                      
    %memref1 = memref.alloc() : memref<5x5xf32>
    bufferization.materialize_in_destination %2 in %memref1 {writable} : (tensor<5x5xf32>, memref<5x5xf32>) -> ()
    %cast1 = memref.cast %memref1 : memref<5x5xf32> to memref<*xf32>

    call @printMemrefF32(%cast1) : (memref<*xf32>) -> ()
    memref.dealloc %memref1 : memref<5x5xf32>
    return
  }

  func.func @test_sub() {
    %1 = tensor.empty() : tensor<5x5xf32>
    
    %denseA = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}} 
        dense<[[ 1.0,  2.0,  3.0,  0.0,  0.0],
               [ 4.0,  5.0,  6.0,  7.0,  0.0],
               [ 0.0,  8.0,  9.0, 10.0, 11.0],
               [ 0.0,  0.0, 12.0, 13.0, 14.0],
               [ 0.0,  0.0,  0.0, 15.0, 16.0]]> : tensor<5x5xf32>

    %diaB = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[ 0.0,  0.0,  6.0, 10.0, 14.0],
               [ 0.0,  3.0,  7.0, 11.0, 15.0],
               [ 1.0,  4.0,  8.0, 12.0, 16.0],
               [ 2.0,  5.0,  9.0, 13.0,  0.0]]> : tensor<4x5xf32>

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [5, 5]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}0, 0, 3, 0, 0],
    // CHECK-NEXT:  [1, 1, 1, 7, 0],
    // CHECK-NEXT:  [-6, 1, 1, 1, 11],
    // CHECK-NEXT:  [0, -10, 1, 1, 1],
    // CHECK-NEXT:  [0, 0, -14, 0, 0]]
    %3 = dia.elementwise kind = <sub> ins(%denseA, %diaB : tensor<5x5xf32>, tensor<4x5xf32>) 
                                      outs(%1 : tensor<5x5xf32>) -> tensor<5x5xf32>
                                      
    %memref2 = memref.alloc() : memref<5x5xf32>
    bufferization.materialize_in_destination %3 in %memref2 {writable} : (tensor<5x5xf32>, memref<5x5xf32>) -> ()
    %cast2 = memref.cast %memref2 : memref<5x5xf32> to memref<*xf32>

    call @printMemrefF32(%cast2) : (memref<*xf32>) -> ()
    memref.dealloc %memref2 : memref<5x5xf32>
    return
  }

  func.func @test_batch() {
    %0 = tensor.empty() : tensor<2x5x5xf32>
    
    %denseA = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} 
        dense<[[[ 1.0,  2.0,  3.0,  0.0,  0.0],
                [ 4.0,  5.0,  6.0,  7.0,  0.0],
                [ 0.0,  8.0,  9.0, 10.0, 11.0],
                [ 0.0,  0.0, 12.0, 13.0, 14.0],
                [ 0.0,  0.0,  0.0, 15.0, 16.0]],
               [[ 2.0,  4.0,  6.0,  0.0,  0.0],
                [ 8.0, 10.0, 12.0, 14.0,  0.0],
                [ 0.0, 16.0, 18.0, 20.0, 22.0],
                [ 0.0,  0.0, 24.0, 26.0, 28.0],
                [ 0.0,  0.0,  0.0, 30.0, 32.0]]]> : tensor<2x5x5xf32>

    %diaB = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} 
        dense<[[[ 0.0,  0.0,  6.0, 10.0, 14.0],
                [ 0.0,  3.0,  7.0, 11.0, 15.0],
                [ 1.0,  4.0,  8.0, 12.0, 16.0],
                [ 2.0,  5.0,  9.0, 13.0,  0.0]],
               [[ 0.0,  0.0, 12.0, 20.0, 28.0],
                [ 0.0,  6.0, 14.0, 22.0, 30.0],
                [ 2.0,  8.0, 16.0, 24.0, 32.0],
                [ 4.0, 10.0, 18.0, 26.0,  0.0]]]> : tensor<2x4x5xf32>

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [2, 5, 5]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[\[}}2, 4, 3, 0, 0],
    // CHECK-NEXT:  [7, 9, 11, 7, 0],
    // CHECK-NEXT:  [6, 15, 17, 19, 11],
    // CHECK-NEXT:  [0, 10, 23, 25, 27],
    // CHECK-NEXT:  [0, 0, 14, 30, 32]],
    // CHECK-NEXT:  {{\[\[}}4, 8, 6, 0, 0],
    // CHECK-NEXT:  [14, 18, 22, 14, 0],
    // CHECK-NEXT:  [12, 30, 34, 38, 22],
    // CHECK-NEXT:  [0, 20, 46, 50, 54],
    // CHECK-NEXT:  [0, 0, 28, 60, 64]]]
    %add_res = dia.elementwise kind = <add> ins(%denseA, %diaB : tensor<2x5x5xf32>, tensor<2x4x5xf32>) 
                                      outs(%0 : tensor<2x5x5xf32>) -> tensor<2x5x5xf32>
                                      
    %memref_add = memref.alloc() : memref<2x5x5xf32>
    bufferization.materialize_in_destination %add_res in %memref_add {writable} : (tensor<2x5x5xf32>, memref<2x5x5xf32>) -> ()
    %cast_add = memref.cast %memref_add : memref<2x5x5xf32> to memref<*xf32>

    call @printMemrefF32(%cast_add) : (memref<*xf32>) -> ()
    memref.dealloc %memref_add : memref<2x5x5xf32>

    return
  }

  func.func @main() {
    call @test_add() : () -> ()
    call @test_sub() : () -> ()
    call @test_batch() : () -> ()
    return
  }
}
