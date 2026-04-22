// RUN: %build/tools/alg-opt %s \
// RUN:   --banded-analysis \
// RUN:   --banded-rewrite \
// RUN:   --reconcile-unrealized-casts \
// RUN:   --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:   --convert-linalg-to-loops \
// RUN:   --convert-scf-to-cf \
// RUN:   --convert-cf-to-llvm \
// RUN:   --convert-math-to-llvm \
// RUN:   --convert-math-to-libm \
// RUN:   --expand-strided-metadata \
// RUN:   --lower-affine \
// RUN:   --convert-arith-to-llvm \
// RUN:   --finalize-memref-to-llvm \
// RUN:   --convert-func-to-llvm \
// RUN:   --reconcile-unrealized-casts | \
// RUN:  mlir-runner \
// RUN:     --entry-point-result=void \
// RUN:     --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format --shared-libs=%llvm_root/build/lib/libmlir_c_runner_utils.%lib_format > %t
// RUN:  FileCheck %s < %t

// CHECK:      rank = 2 offset = 0 sizes = [5, 5] strides = [5, 1]
// CHECK-NEXT: {{[[][[]}}0.5,{{.*}}0.5,{{.*}}0,{{.*}}0,{{.*}}0],
// CHECK-NEXT: {{[[:space:]]}}[0.333333,{{.*}}0.333333,{{.*}}0.333333,{{.*}}0,{{.*}}0],
// CHECK-NEXT: {{[[:space:]]}}[0,{{.*}}0.333333,{{.*}}0.333333,{{.*}}0.333333,{{.*}}0],
// CHECK-NEXT: {{[[:space:]]}}[0,{{.*}}0,{{.*}}0.333333,{{.*}}0.333333,{{.*}}0.333333],
// CHECK-NEXT: {{[[:space:]]}}[0,{{.*}}0,{{.*}}0,{{.*}}0.5,{{.*}}0.5]]

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() {
    %0 = arith.constant
        {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<5x5xf32>

    %1 = dia.softmax (%0 : tensor<5x5xf32>) -> tensor<5x5xf32>

    %memref = memref.alloc() : memref<5x5xf32>
    bufferization.materialize_in_destination %1 in %memref {writable}
        : (tensor<5x5xf32>, memref<5x5xf32>) -> ()
    %cast = memref.cast %memref : memref<5x5xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<5x5xf32>
    return
  }
}
