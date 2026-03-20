// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.so > %t
// RUN: FileCheck %s < %t

module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() {
    %0 = tensor.empty() : tensor<3x5xf32>
    
    %denseA = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 2 : i64, propertyDims = [0, 1]}} 
        dense<[[ 1.0,  2.0,  3.0,  0.0,  0.0],
               [ 4.0,  1.0,  2.0,  3.0,  0.0],
               [ 0.0,  4.0,  1.0,  2.0,  3.0],
               [ 0.0,  0.0,  4.0,  1.0,  2.0],
               [ 0.0,  0.0,  0.0,  4.0,  1.0]]> : tensor<5x5xf32>

    %diaB = arith.constant {metadata = {dia = true, lowerBw = 2 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}} 
        dense<[[ 0.0, 0.0, 6.0, 10.0, 14.0], // L2: 3 elements
               [ 0.0, 3.0,  7.0, 11.0, 15.0], // L1: 4 elements
               [ 1.0,  4.0,  8.0, 12.0, 16.0], // M0: 5 elements
               [ 2.0,  5.0,  9.0, 13.0, 0.0 ]]> : tensor<4x5xf32> // U1: 4 elements

    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [3, 5]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}0, 12, 28, 44, 60],
    // CHECK-NEXT:  [1, 4, 8, 12, 16],
    // CHECK-NEXT:  [4, 10, 18, 26, 0]]
    %mul_res = dia.elementwise kind = <mul> ins(%denseA, %diaB : tensor<5x5xf32>, tensor<4x5xf32>) 
                                            outs(%0 : tensor<3x5xf32>) -> tensor<3x5xf32>
                                      
    %memref = memref.alloc() : memref<3x5xf32>
    bufferization.materialize_in_destination %mul_res in %memref {writable} : (tensor<3x5xf32>, memref<3x5xf32>) -> ()
    %cast = memref.cast %memref : memref<3x5xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<3x5xf32>

    return
  }
}
