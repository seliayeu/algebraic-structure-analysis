// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format> %t
// RUN: FileCheck %s < %t

#map  = affine_map<(d0) -> (0, d0)>
#map1 = affine_map<(d0) -> (d0, d0)>

module {
  func.func private @printMemrefF32(memref<*xf32>)
 
  func.func @main() { 
    %dia1 = arith.constant {metadata = {dia = true, upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 3.0]]> : tensor<1x3xf32>
    %dense = arith.constant {metadata = {upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 0.0],
               [0.0, 2.0, 0.0],
               [0.0, 0.0, 3.0]]> : tensor<3x3xf32>
    %0 = tensor.empty() : tensor<1x3xf32>
    %1 = dia.matmul ins(%dia1, %dense : tensor<1x3xf32>, tensor<3x3xf32>)
                    outs(%0 : tensor<1x3xf32>) -> tensor<1x3xf32>

    // CHECK: [1, 4, 9]
    %memref = memref.alloc() : memref<1x3xf32>
    bufferization.materialize_in_destination %1 in %memref {writable}
        : (tensor<1x3xf32>, memref<1x3xf32>) -> ()
    %cast = memref.cast %memref : memref<1x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<1x3xf32>

    return
  }
}
