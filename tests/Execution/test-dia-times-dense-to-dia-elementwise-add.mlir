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
  
  func.func @main() {
    %0 = tensor.empty() : tensor<5x5xf32>
    
    %diaA = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[ 0.0, 0.0, 6.0, 10.0, 14.0], // L2
               [ 0.0, 3.0,  7.0, 11.0, 15.0], // L1
               [ 1.0,  4.0,  8.0, 12.0, 16.0], // M0
               [ 2.0,  5.0,  9.0, 13.0,  0.0]]> : tensor<4x5xf32> // U1
               
    %denseB = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}} 
        dense<[[ 1.0,  2.0,  3.0,  0.0,  0.0],
               [ 4.0,  1.0,  2.0,  3.0,  0.0],
               [ 0.0,  4.0,  1.0,  2.0,  3.0],
               [ 0.0,  0.0,  4.0,  1.0,  2.0],
               [ 0.0,  0.0,  0.0,  4.0,  1.0]]> : tensor<5x5xf32>

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [5, 5]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}0, 0, 6, 10, 14],
    // CHECK-NEXT:  [0, 7, 11, 15, 19],
    // CHECK-NEXT:  [2, 5, 9, 13, 17],
    // CHECK-NEXT:  [4, 7, 11, 15, 0],
    // CHECK-NEXT:  [3, 3, 3, 0, 0]]
    %add_res = dia.elementwise kind = <add> ins(%diaA, %denseB : tensor<4x5xf32>, tensor<5x5xf32>) 
                                            outs(%0 : tensor<5x5xf32>) -> tensor<5x5xf32>
                                      
    %memref = memref.alloc() : memref<5x5xf32>
    bufferization.materialize_in_destination %add_res in %memref {writable} : (tensor<5x5xf32>, memref<5x5xf32>) -> ()
    %cast = memref.cast %memref : memref<5x5xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<5x5xf32>

    return
  }
}
