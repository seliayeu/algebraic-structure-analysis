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

  // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [2, 4]
  // CHECK: {{\[}}[0{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}],
  // CHECK: [2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}]]
  func.func @l2u0_l2u0_forced() {
    %I0 = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<4x4xf32>
    %I1 = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<2.0> : tensor<4x4xf32>
    %e1 = arith.constant dense<0.0> : tensor<2x4xf32>

    %R0 = dia.matmul ins(%I0, %I1 : tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%e1 : tensor<2x4xf32>) -> tensor<2x4xf32> {metadata = {lowerBw = 1: i64, propertyDims = [0, 1], upperBw = 1: i64}}

    %memref = memref.alloc() : memref<2x4xf32>
    bufferization.materialize_in_destination %R0 in %memref {writable}
        : (tensor<2x4xf32>, memref<2x4xf32>) -> ()
    %cast = memref.cast %memref : memref<2x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<2x4xf32>
    return
  }

  // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [4, 4]
  // CHECK: {{\[}}[0{{.*}}, 0{{.*}}, 0{{.*}}, 4{{.*}}],
  // CHECK: [0{{.*}}, 0{{.*}}, 6{{.*}}, 6{{.*}}],
  // CHECK: [0{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}],
  // CHECK: [2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}]]
  func.func @l2u0_l2u0() {
    %I0 = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<4x4xf32>
    %I1 = arith.constant {metadata = {lowerBw = 2 : i64, upperBw = 0 : i64, propertyDims = [0, 1]}}
        dense<2.0> : tensor<4x4xf32>
    %e1 = arith.constant dense<0.0> : tensor<4x4xf32>

    %R0 = dia.matmul ins(%I0, %I1 : tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%e1 : tensor<4x4xf32>) -> tensor<4x4xf32>

    %memref = memref.alloc() : memref<4x4xf32>
    bufferization.materialize_in_destination %R0 in %memref {writable}
        : (tensor<4x4xf32>, memref<4x4xf32>) -> ()
    %cast = memref.cast %memref : memref<4x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<4x4xf32>
    return
  }

  // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [7, 4]
  // CHECK: {{\[}}[0{{.*}}, 0{{.*}}, 0{{.*}}, 4{{.*}}],
  // CHECK: [0{{.*}}, 0{{.*}}, 6{{.*}}, 4{{.*}}],
  // CHECK: [0{{.*}}, 8{{.*}}, 6{{.*}}, 4{{.*}}],
  // CHECK: [8{{.*}}, 8{{.*}}, 6{{.*}}, 4{{.*}}],
  // CHECK: [8{{.*}}, 6{{.*}}, 4{{.*}}, 0{{.*}}],
  // CHECK: [6{{.*}}, 4{{.*}}, 0{{.*}}, 0{{.*}}],
  // CHECK: [4{{.*}}, 0{{.*}}, 0{{.*}}, 0{{.*}}]]
  func.func @l1u3_l3u1() {
    %I0 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 3 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<4x4xf32>
    %I1 = arith.constant {metadata = {lowerBw = 3 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<2.0> : tensor<4x4xf32>
    %e1 = arith.constant dense<0.0> : tensor<7x4xf32>

    %R0 = dia.matmul ins(%I0, %I1 : tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%e1 : tensor<7x4xf32>) -> tensor<7x4xf32>

    %memref = memref.alloc() : memref<7x4xf32>
    bufferization.materialize_in_destination %R0 in %memref {writable}
        : (tensor<7x4xf32>, memref<7x4xf32>) -> ()
    %cast = memref.cast %memref : memref<7x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<7x4xf32>
    return
  }

  // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [3, 4]
  // CHECK: {{\[}}[0{{.*}}, 8{{.*}}, 6{{.*}}, 4{{.*}}],
  // CHECK: [8{{.*}}, 8{{.*}}, 6{{.*}}, 4{{.*}}],
  // CHECK: [8{{.*}}, 6{{.*}}, 4{{.*}}, 0{{.*}}]]
  func.func @l1u3_l3u1_forced() {
    %I0 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 3 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<4x4xf32>
    %I1 = arith.constant {metadata = {lowerBw = 3 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<2.0> : tensor<4x4xf32>
    %e1 = arith.constant dense<0.0> : tensor<3x4xf32>

    %R0 = dia.matmul ins(%I0, %I1 : tensor<4x4xf32>, tensor<4x4xf32>)
                     outs(%e1 : tensor<3x4xf32>) -> tensor<3x4xf32> {metadata = {lowerBw = 1: i64, propertyDims = [0, 1], upperBw = 1: i64}}

    %memref = memref.alloc() : memref<3x4xf32>
    bufferization.materialize_in_destination %R0 in %memref {writable}
        : (tensor<3x4xf32>, memref<3x4xf32>) -> ()
    %cast = memref.cast %memref : memref<3x4xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<3x4xf32>
    return
  }

  // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [1, 10]
  // CHECK: {{\[}}[4{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 4{{.*}}]]
  func.func @diagonal() {
    %I0 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<10x10xf32>
    %I1 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<2.0> : tensor<10x10xf32>

    %e1 = arith.constant dense<0.0> : tensor<1x10xf32>

    %R0 = dia.matmul ins(%I0, %I1 : tensor<10x10xf32>, tensor<10x10xf32>)
                     outs(%e1 : tensor<1x10xf32>) -> tensor<1x10xf32> {metadata = {lowerBw = 0: i64, propertyDims = [0, 1], upperBw = 0: i64}}

    %memref = memref.alloc() : memref<1x10xf32>
    bufferization.materialize_in_destination %R0 in %memref {writable}
        : (tensor<1x10xf32>, memref<1x10xf32>) -> ()
    %cast = memref.cast %memref : memref<1x10xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<1x10xf32>
    return
  }

  // CHECK: Unranked Memref {{.*}} rank = 2 offset = 0 sizes = [5, 10]
  // CHECK: {{\[}}[0{{.*}}, 0{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}],
  // CHECK-NEXT: [0{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}],
  // CHECK-NEXT: [4{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 6{{.*}}, 4{{.*}}],
  // CHECK-NEXT: [4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 4{{.*}}, 0{{.*}}],
  // CHECK-NEXT: [2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 2{{.*}}, 0{{.*}}, 0{{.*}}]]
  func.func @regular() {
    %I0 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<1.0> : tensor<10x10xf32>
    %I1 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [0, 1]}}
        dense<2.0> : tensor<10x10xf32>
    %e1 = arith.constant dense<0.0> : tensor<5x10xf32>

    %R0 = dia.matmul  ins(%I0, %I1 : tensor<10x10xf32>, tensor<10x10xf32>)
                     outs(%e1 : tensor<5x10xf32>) -> tensor<5x10xf32>

    %memref = memref.alloc() : memref<5x10xf32>
    bufferization.materialize_in_destination %R0 in %memref {writable}
        : (tensor<5x10xf32>, memref<5x10xf32>) -> ()
    %cast = memref.cast %memref : memref<5x10xf32> to memref<*xf32>
    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
    memref.dealloc %memref : memref<5x10xf32>
    return
  }

  func.func @main() {
    call @l2u0_l2u0_forced() : () -> ()
    call @l2u0_l2u0() : () -> ()
    call @l1u3_l3u1() : () -> ()
    call @l1u3_l3u1_forced() : () -> ()
    call @diagonal() : () -> ()
    call @regular() : () -> ()
    return
  }

}
