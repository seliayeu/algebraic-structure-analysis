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
    %0 = tensor.empty() : tensor<3x3xf32>
    %dia1 = arith.constant {metadata = {dia = true, upperBw = 0 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 1.0, 2.0],
               [1.0, 2.0, 3.0]]> : tensor<2x3xf32>
    %dia2 = arith.constant {metadata = {dia = true, upperBw = 0 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 1.0, 2.0],
               [1.0, 2.0, 3.0]]> : tensor<2x3xf32>
    // CHECK:      Unranked Memref
    // CHECK-SAME: sizes = [3, 3]
    // CHECK-SAME: data =
    // CHECK-NEXT: {{\[\[}}0, 0, 2],
    // CHECK-NEXT: [0, 3, 10]
    // CHECK-NEXT: [1, 4, 9]]
    %1 = dia.matmul ins(%dia1, %dia2 : tensor<2x3xf32>, tensor<2x3xf32>)
                    outs(%0 : tensor<3x3xf32>) -> tensor<3x3xf32>
    %memref = memref.alloc() : memref<3x3xf32>
    bufferization.materialize_in_destination %1 in %memref {writable} : (tensor<3x3xf32>, memref<3x3xf32>) -> ()
    %cast = memref.cast %memref : memref<3x3xf32> to memref<*xf32>

    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<3x3xf32>
    return
  }
}
