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
    %0 = tensor.empty() : tensor<5x4xf32>
    
    %diaA = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[0.0, 0.0, 1.0, 2.0, 3.0],      // L2: 3 elements
               [0.0, 4.0, 5.0, 6.0, 7.0],      // L1: 4 elements
               [8.0, 9.0, 10.0, 11.0, 0.0],    // M0: 4 elements
               [12.0, 13.0, 14.0, 0.0, 0.0]]> : tensor<4x5xf32> // U1: 3 elements
               
    %diaB = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}} 
        dense<[[0.0, 1.0, 1.0, 1.0, 1.0],      // L1: 4 elements
               [2.0, 2.0, 2.0, 2.0, 0.0],      // M0: 4 elements
               [3.0, 3.0, 3.0, 0.0, 0.0],      // U1: 3 elements
               [4.0, 4.0, 0.0, 0.0, 0.0]]> : tensor<4x5xf32> // U2: 2 elements

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [5, 4]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}10, 15, 4, 0],
    // CHECK-NEXT:  [5, 11, 16, 4],
    // CHECK-NEXT:  [1, 6, 12, 17],
    // CHECK-NEXT:  [0, 2, 7, 13],
    // CHECK-NEXT:  [0, 0, 3, 8]]
    %1 = dia.elementwise kind = <add> ins(%diaA, %diaB : tensor<4x5xf32>, tensor<4x5xf32>) 
                                      outs(%0 : tensor<5x4xf32>) -> tensor<5x4xf32>
                                      
    %memref = memref.alloc() : memref<5x4xf32>
    bufferization.materialize_in_destination %1 in %memref {writable} : (tensor<5x4xf32>, memref<5x4xf32>) -> ()
    %cast = memref.cast %memref : memref<5x4xf32> to memref<*xf32>

    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<5x4xf32>
    return
  }

  func.func @test_sub() {
    %0 = tensor.empty() : tensor<5x4xf32>
    
    %diaA = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[0.0, 0.0, 1.0, 2.0, 3.0], 
               [0.0, 4.0, 5.0, 6.0, 7.0], 
               [8.0, 9.0, 10.0, 11.0, 0.0], 
               [12.0, 13.0, 14.0, 0.0, 0.0]]> : tensor<4x5xf32> 
               
    %diaB = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}} 
        dense<[[0.0, 1.0, 1.0, 1.0, 1.0], 
               [2.0, 2.0, 2.0, 2.0, 0.0], 
               [3.0, 3.0, 3.0, 0.0, 0.0], 
               [4.0, 4.0, 0.0, 0.0, 0.0]]> : tensor<4x5xf32> 

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [5, 4]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}6, 9, -4, 0],
    // CHECK-NEXT:  [3, 7, 10, -4],
    // CHECK-NEXT:  [1, 4, 8, 11],
    // CHECK-NEXT:  [0, 2, 5, 9],
    // CHECK-NEXT:  [0, 0, 3, 6]]
    %1 = dia.elementwise kind = <sub> ins(%diaA, %diaB : tensor<4x5xf32>, tensor<4x5xf32>) 
                                      outs(%0 : tensor<5x4xf32>) -> tensor<5x4xf32>
                                      
    %memref = memref.alloc() : memref<5x4xf32>
    bufferization.materialize_in_destination %1 in %memref {writable} : (tensor<5x4xf32>, memref<5x4xf32>) -> ()
    %cast = memref.cast %memref : memref<5x4xf32> to memref<*xf32>

    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<5x4xf32>
    return
  }

  func.func @test_batch() {
    %0 = tensor.empty() : tensor<2x5x4xf32>
    
    // Batch 0: Original values. Batch 1: Original values * 2.
    // Note: propertyDims updated to [1, 2] to account for the leading batch dimension.
    %diaA = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} 
        dense<[[[0.0, 0.0, 1.0, 2.0, 3.0], 
                [0.0, 4.0, 5.0, 6.0, 7.0], 
                [8.0, 9.0, 10.0, 11.0, 0.0], 
                [12.0, 13.0, 14.0, 0.0, 0.0]],
               [[0.0, 0.0, 2.0, 4.0, 6.0], 
                [0.0, 8.0, 10.0, 12.0, 14.0], 
                [16.0, 18.0, 20.0, 22.0, 0.0], 
                [24.0, 26.0, 28.0, 0.0, 0.0]]]> : tensor<2x4x5xf32>
               
    %diaB = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [1, 2]}} 
        dense<[[[0.0, 1.0, 1.0, 1.0, 1.0], 
                [2.0, 2.0, 2.0, 2.0, 0.0], 
                [3.0, 3.0, 3.0, 0.0, 0.0], 
                [4.0, 4.0, 0.0, 0.0, 0.0]],
               [[0.0, 2.0, 2.0, 2.0, 2.0], 
                [4.0, 4.0, 4.0, 4.0, 0.0], 
                [6.0, 6.0, 6.0, 0.0, 0.0], 
                [8.0, 8.0, 0.0, 0.0, 0.0]]]> : tensor<2x4x5xf32>

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [2, 5, 4]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[\[}}10, 15, 4, 0],
    // CHECK-NEXT:  [5, 11, 16, 4],
    // CHECK-NEXT:  [1, 6, 12, 17],
    // CHECK-NEXT:  [0, 2, 7, 13],
    // CHECK-NEXT:  [0, 0, 3, 8]],
    // CHECK-NEXT:  {{\[\[}}20, 30, 8, 0],
    // CHECK-NEXT:  [10, 22, 32, 8],
    // CHECK-NEXT:  [2, 12, 24, 34],
    // CHECK-NEXT:  [0, 4, 14, 26],
    // CHECK-NEXT:  [0, 0, 6, 16]]]
    %1 = dia.elementwise kind = <add> ins(%diaA, %diaB : tensor<2x4x5xf32>, tensor<2x4x5xf32>) 
                                      outs(%0 : tensor<2x5x4xf32>) -> tensor<2x5x4xf32>
                                      
    %memref = memref.alloc() : memref<2x5x4xf32>
    bufferization.materialize_in_destination %1 in %memref {writable} : (tensor<2x5x4xf32>, memref<2x5x4xf32>) -> ()
    %cast = memref.cast %memref : memref<2x5x4xf32> to memref<*xf32>

    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x5x4xf32>
    return
  }

  func.func @main() {
    call @test_add() : () -> ()
    call @test_sub() : () -> ()
    call @test_batch() : () -> ()
    return
  }
}
