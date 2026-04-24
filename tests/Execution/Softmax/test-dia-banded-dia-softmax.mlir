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

// CHECK:      rank = 2 offset = 0 sizes = [7, 4] strides = [4, 1]
// CHECK-NEXT: {{[[][[]}}0,{{.*}}0,{{.*}}0,{{.*}}0.25],
// CHECK-NEXT: {{[[:space:]]}}[0,{{.*}}0,{{.*}}0.174878,{{.*}}0.25],
// CHECK-NEXT: {{[[:space:]]}}[0,{{.*}}0.365529,{{.*}}0.174878,{{.*}}0.25],
// CHECK-NEXT: {{[[:space:]]}}[0.870049,{{.*}}0.365529,{{.*}}0.174878,{{.*}}0.25],
// CHECK-NEXT: {{[[:space:]]}}[0.0433172,{{.*}}0.134471,{{.*}}0.475367,{{.*}}0],
// CHECK-NEXT: {{[[:space:]]}}[0.0433172,{{.*}}0.134471,{{.*}}0,{{.*}}0],
// CHECK-NEXT: {{[[:space:]]}}[0.0433172,{{.*}}0,{{.*}}0,{{.*}}0]]

module {
  func.func private @printMemrefF32(memref<*xf32>)

  func.func @main() {
    %0 = arith.constant
        {metadata = {dia = true, lowerBw = 3 : i64, upperBw = 3 : i64, propertyDims = [0, 1]}}
          dense<[[0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0, 1.0],
                [0.0, 3.0, 2.0, 1.0],
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 2.0, 3.0, 0.0],
                [1.0, 2.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0]]> : tensor<7x4xf32>

    %1 = dia.softmax (%0 : tensor<7x4xf32>) -> tensor<7x4xf32>

    %memref = memref.alloc() : memref<7x4xf32>
    bufferization.materialize_in_destination %1 in %memref {writable}
        : (tensor<7x4xf32>, memref<7x4xf32>) -> ()
    %cast = memref.cast %memref : memref<7x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<7x4xf32>
    return
  }
}

