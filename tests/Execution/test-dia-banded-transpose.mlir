// RUN: %build/tools/alg-opt %s --banded-analysis="detect-dia=true" --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format > %t
// RUN: FileCheck %s < %t

// CHECK: Unranked Memref base@ = {{.*}} rank = 2 offset = 0 sizes = [4, 3]
// CHECK: {{\[}}[0{{.*}}, 0{{.*}}, 3{{.*}}],
// CHECK-NEXT: [0{{.*}}, 2{{.*}}, 6{{.*}}],
// CHECK-NEXT: [1{{.*}}, 5{{.*}}, 9{{.*}}],
// CHECK-NEXT: [4{{.*}}, 8{{.*}}, 0{{.*}}]]

module {
  func.func private @printMemrefF32(memref<*xf32>)
  func.func @main() {
    %dia = arith.constant {metadata = {upperBw = 2 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 4.0, 8.0],
               [1.0, 5.0, 9.0],
               [2.0, 6.0, 0.0],
               [3.0, 0.0, 0.0]]> : tensor<4x3xf32>
    %transposed = dia.transpose (%dia: tensor<4x3xf32>)
    %memref = memref.alloc() : memref<4x3xf32>
    bufferization.materialize_in_destination %transposed in %memref {writable}
        : (tensor<4x3xf32>, memref<4x3xf32>) -> ()
    %cast = memref.cast %memref : memref<4x3xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<4x3xf32>
    return
  }
}
