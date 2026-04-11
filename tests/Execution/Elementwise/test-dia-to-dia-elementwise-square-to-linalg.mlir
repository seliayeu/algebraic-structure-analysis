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
  
  func.func @test_square() {
    %0 = tensor.empty() : tensor<2x4xf32>
    
    %diaA = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}} 
        dense<[[1.0, 1.0, 1.0, 0.0], 
               [2.0, 2.0, 2.0, 2.0]]> : tensor<2x4xf32>

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [2, 4]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}1, 1, 1, 0],
    // CHECK-NEXT:  [4, 4, 4, 4]]
    %1 = dia.elementwise kind = <square> ins(%diaA : tensor<2x4xf32>) 
                                         outs(%0 : tensor<2x4xf32>) -> tensor<2x4xf32>
                                        
    %memref = memref.alloc() : memref<2x4xf32>
    bufferization.materialize_in_destination %1 in %memref {writable} : (tensor<2x4xf32>, memref<2x4xf32>) -> ()
    %cast = memref.cast %memref : memref<2x4xf32> to memref<*xf32>

    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x4xf32>
    
    return
  }

  func.func @test_batch() {
    %0 = tensor.empty() : tensor<2x2x4xf32>
    
    %diaA = arith.constant {metadata = {dia = true, lowerBw = 1 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} 
        dense<[[[1.0, 1.0, 1.0, 0.0], 
                [2.0, 2.0, 2.0, 2.0]],
               [[3.0, 3.0, 3.0, 0.0], 
                [6.0, 6.0, 6.0, 6.0]]]> : tensor<2x2x4xf32>

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [2, 2, 4]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[\[}}1, 1, 1, 0],
    // CHECK-NEXT:  [4, 4, 4, 4]],
    // CHECK-NEXT:  {{\[\[}}9, 9, 9, 0],
    // CHECK-NEXT:  [36, 36, 36, 36]]]
    %1 = dia.elementwise kind = <square> ins(%diaA : tensor<2x2x4xf32>) 
                                         outs(%0 : tensor<2x2x4xf32>) -> tensor<2x2x4xf32>
                                        
    %memref = memref.alloc() : memref<2x2x4xf32>
    bufferization.materialize_in_destination %1 in %memref {writable} : (tensor<2x2x4xf32>, memref<2x2x4xf32>) -> ()
    %cast = memref.cast %memref : memref<2x2x4xf32> to memref<*xf32>

    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x2x4xf32>
    
    return
  }

  func.func @main() {
    call @test_square() : () -> ()
    call @test_batch() : () -> ()
    return
  }
}
