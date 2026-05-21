func.func @kernel() -> f32 {
  %c0 = arith.constant 0 : index
  %cf1 = arith.constant 1.0 : f32

  %e1 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e2 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e3 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e4 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e5 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e6 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e7 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e8 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e9 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e10 = tensor.empty(%c0) : tensor<?x1024xf32>

  %input_empty = tensor.empty() : tensor<Yx1024xf32>
  %W1_empty = tensor.empty() : tensor<Yx1024xf32>
  %W2_empty = tensor.empty() : tensor<Yx1024xf32>
  %W3_empty = tensor.empty() : tensor<Yx1024xf32>
  %W4_empty = tensor.empty() : tensor<Yx1024xf32>
  %W5_empty = tensor.empty() : tensor<Yx1024xf32>
  %W6_empty = tensor.empty() : tensor<Yx1024xf32>
  %W7_empty = tensor.empty() : tensor<Yx1024xf32>
  %W8_empty = tensor.empty() : tensor<Yx1024xf32>
  %W9_empty = tensor.empty() : tensor<Yx1024xf32>
  %W10_empty = tensor.empty() : tensor<Yx1024xf32>

  %input = linalg.fill {metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%input_empty : tensor<Yx1024xf32>) -> tensor<Yx1024xf32>

  %W1 = linalg.fill {metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W1_empty : tensor<Yx1024xf32>) -> tensor<Yx1024xf32>
  %W2 = linalg.fill {metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W2_empty : tensor<Yx1024xf32>) -> tensor<Yx1024xf32>
  %W3 = linalg.fill {metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W3_empty : tensor<Yx1024xf32>) -> tensor<Yx1024xf32>
  %W4 = linalg.fill {metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W4_empty : tensor<Yx1024xf32>) -> tensor<Yx1024xf32>
  %W5 = linalg.fill {metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W5_empty : tensor<Yx1024xf32>) -> tensor<Yx1024xf32>
  %W6 = linalg.fill {metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W6_empty : tensor<Yx1024xf32>) -> tensor<Yx1024xf32>
  %W7 = linalg.fill {metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W7_empty : tensor<Yx1024xf32>) -> tensor<Yx1024xf32>
  %W8 = linalg.fill {metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W8_empty : tensor<Yx1024xf32>) -> tensor<Yx1024xf32>
  %W9 = linalg.fill {metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W9_empty : tensor<Yx1024xf32>) -> tensor<Yx1024xf32>
  %W10 = linalg.fill {metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W10_empty : tensor<Yx1024xf32>) -> tensor<Yx1024xf32>

  %m1 = dia.matmul ins(%input, %W1 : tensor<Yx1024xf32>, tensor<Yx1024xf32>) outs(%e1 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m2 = dia.matmul ins(%m1, %W2 : tensor<?x1024xf32>, tensor<Yx1024xf32>) outs(%e2 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m3 = dia.matmul ins(%m2, %W3 : tensor<?x1024xf32>, tensor<Yx1024xf32>) outs(%e3 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m4 = dia.matmul ins(%m3, %W4 : tensor<?x1024xf32>, tensor<Yx1024xf32>) outs(%e4 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m5 = dia.matmul ins(%m4, %W5 : tensor<?x1024xf32>, tensor<Yx1024xf32>) outs(%e5 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m6 = dia.matmul ins(%m5, %W6 : tensor<?x1024xf32>, tensor<Yx1024xf32>) outs(%e6 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m7 = dia.matmul ins(%m6, %W7 : tensor<?x1024xf32>, tensor<Yx1024xf32>) outs(%e7 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m8 = dia.matmul ins(%m7, %W8 : tensor<?x1024xf32>, tensor<Yx1024xf32>) outs(%e8 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m9 = dia.matmul ins(%m8, %W9 : tensor<?x1024xf32>, tensor<Yx1024xf32>) outs(%e9 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m10 = dia.matmul ins(%m9, %W10 : tensor<?x1024xf32>, tensor<Yx1024xf32>) outs(%e10 : tensor<?x1024xf32>) -> tensor<?x1024xf32>

  %index = arith.constant 0 : index
  %result = tensor.extract %m10[%index, %index] : tensor<?x1024xf32>

  return %result : f32
}
