// RUN: %build/tools/alg-opt %s --banded-analysis --banded-rewrite \
// RUN:  --reconcile-unrealized-casts \
// RUN:  --one-shot-bufferize="allow-return-allocs-from-loops=true bufferize-function-boundaries=true" \
// RUN:  --convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm \
// RUN:  --convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm \
// RUN:  --reconcile-unrealized-casts | \
// RUN:  mlir-runner --entry-point-result=void \
// RUN:  --shared-libs=%llvm_root/build/lib/libmlir_runner_utils.%lib_format > %t
// RUN: FileCheck %s < %t

func.func private @printMemrefF32(memref<*xf32>)

func.func @main() {
  %c0 = arith.constant 0 : index

  %e1 = tensor.empty(%c0) : tensor<4x?x4xf32>
  %e2 = tensor.empty(%c0) : tensor<4x?x4xf32>
  %e3 = tensor.empty(%c0) : tensor<4x?x4xf32>
  %e4 = tensor.empty(%c0) : tensor<4x?x4xf32>
  %e5 = tensor.empty(%c0) : tensor<4x?x4xf32>
  %e6 = tensor.empty(%c0) : tensor<4x?x4xf32>
  %e7 = tensor.empty(%c0) : tensor<4x?x4xf32>
  %e8 = tensor.empty(%c0) : tensor<4x?x4xf32>
  %e9 = tensor.empty(%c0) : tensor<4x?x4xf32>
  %e10 = tensor.empty(%c0) : tensor<4x?x4xf32>

  %input = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<4x4x4xf32>
  %W1 = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<4x4x4xf32>
  %W2 = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<4x4x4xf32>
  %W3 = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<4x4x4xf32>
  %W4 = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<4x4x4xf32>
  %W5 = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<4x4x4xf32>
  %W6 = arith.constant {metadata = {lowerBw = 0 : i64, upperBw = 0 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<4x4x4xf32>

  %m1 = dia.batch_matmul ins(%input, %W1 : tensor<4x4x4xf32>, tensor<4x4x4xf32>) outs(%e1 : tensor<4x?x4xf32>) -> tensor<4x?x4xf32>
  %m2 = dia.batch_matmul ins(%input, %W2 : tensor<4x4x4xf32>, tensor<4x4x4xf32>) outs(%e2 : tensor<4x?x4xf32>) -> tensor<4x?x4xf32>
  %m3 = dia.batch_matmul ins(%input, %W3 : tensor<4x4x4xf32>, tensor<4x4x4xf32>) outs(%e3 : tensor<4x?x4xf32>) -> tensor<4x?x4xf32>

  %m4 = dia.batch_matmul ins(%m2, %m3 : tensor<4x?x4xf32>, tensor<4x?x4xf32>) outs(%e4 : tensor<4x?x4xf32>) -> tensor<4x?x4xf32>
  %m5 = dia.batch_matmul ins(%m4, %m1 : tensor<4x?x4xf32>, tensor<4x?x4xf32>) outs(%e5 : tensor<4x?x4xf32>) -> tensor<4x?x4xf32>
  %m6 = dia.batch_matmul ins(%m5, %W4 : tensor<4x?x4xf32>, tensor<4x4x4xf32>) outs(%e6 : tensor<4x?x4xf32>) -> tensor<4x?x4xf32>

  %a1 = dia.elementwise kind = <add> ins(%m6, %input : tensor<4x?x4xf32>, tensor<4x4x4xf32>) outs(%e7 : tensor<4x?x4xf32>) -> tensor<4x?x4xf32>

  %m7 = dia.batch_matmul ins(%a1, %W5 : tensor<4x?x4xf32>, tensor<4x4x4xf32>) outs(%e8 : tensor<4x?x4xf32>) -> tensor<4x?x4xf32>
  %m8 = dia.batch_matmul ins(%m7, %W6 : tensor<4x?x4xf32>, tensor<4x4x4xf32>) outs(%e9 : tensor<4x?x4xf32>) -> tensor<4x?x4xf32>

  %a2 = dia.elementwise kind = <add> ins(%m8, %input : tensor<4x?x4xf32>, tensor<4x4x4xf32>) outs(%e10 : tensor<4x?x4xf32>) -> tensor<4x?x4xf32>


  %di = arith.constant 1: index
  %dim = tensor.dim %a2, %di: tensor<4x?x4xf32>
  %memref = memref.alloc(%dim) : memref<4x?x4xf32>
  bufferization.materialize_in_destination %a2 in writable %memref
      : (tensor<4x?x4xf32>, memref<4x?x4xf32>) -> ()
  %cast = memref.cast %memref : memref<4x?x4xf32> to memref<*xf32>
  call @printMemrefF32(%cast) : (memref<*xf32>) -> ()
  memref.dealloc %memref : memref<4x?x4xf32>

  return
}

// CHECK: Unranked Memref base@ = {{.*}} rank = 3 offset = 0 sizes = [4, 1, 4] strides = [4, 4, 1] data = 
// CHECK-NEXT: {{\[\[\[}}3,    3,    3,    3]],
// CHECK-NEXT:   {{\[\[}}3,    3,    3,    3]],
// CHECK-NEXT:   {{\[\[}}3,    3,    3,    3]],
// CHECK-NEXT:   {{\[\[}}3,    3,    3,    3]]]
