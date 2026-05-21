func.func @kernel() -> f32 {
  %cf1 = arith.constant 1.0 : f32
  %e1 = arith.constant dense<0.0> : tensor<1024x1024xf32>
  %e2 = arith.constant dense<0.0> : tensor<1024x1024xf32>
  %e3 = arith.constant dense<0.0> : tensor<1024x1024xf32>
  %e4 = arith.constant dense<0.0> : tensor<1024x1024xf32>
  %e5 = arith.constant dense<0.0> : tensor<1024x1024xf32>
  %e6 = arith.constant dense<0.0> : tensor<1024x1024xf32>
  %e7 = arith.constant dense<0.0> : tensor<1024x1024xf32>
  %e8 = arith.constant dense<0.0> : tensor<1024x1024xf32>
  %e9 = arith.constant dense<0.0> : tensor<1024x1024xf32>
  %e10 = arith.constant dense<0.0> : tensor<1024x1024xf32>
  %input_empty = tensor.empty() : tensor<1024x1024xf32>
  %W1_empty = tensor.empty() : tensor<1024x1024xf32>
  %W2_empty = tensor.empty() : tensor<1024x1024xf32>
  %W3_empty = tensor.empty() : tensor<1024x1024xf32>
  %W4_empty = tensor.empty() : tensor<1024x1024xf32>
  %W5_empty = tensor.empty() : tensor<1024x1024xf32>
  %W6_empty = tensor.empty() : tensor<1024x1024xf32>
  %W7_empty = tensor.empty() : tensor<1024x1024xf32>
  %W8_empty = tensor.empty() : tensor<1024x1024xf32>
  %W9_empty = tensor.empty() : tensor<1024x1024xf32>
  %W10_empty = tensor.empty() : tensor<1024x1024xf32>

  %input = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%input_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W1 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W1_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W2 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W2_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W3 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W3_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W4 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W4_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W5 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W5_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W6 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W6_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W7 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W7_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W8 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W8_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W9 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W9_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W10 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W10_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %m1 = linalg.matmul ins(%input, %W1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %m2 = linalg.matmul ins(%m1, %W2 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %m3 = linalg.matmul ins(%m2, %W3 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e3 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %m4 = linalg.matmul ins(%m3, %W4 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e4 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %m5 = linalg.matmul ins(%m4, %W5 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e5 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %m6 = linalg.matmul ins(%m5, %W6 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e6 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %m7 = linalg.matmul ins(%m6, %W7 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e7 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %m8 = linalg.matmul ins(%m7, %W8 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e8 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %m9 = linalg.matmul ins(%m8, %W9 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e9 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %m10 = linalg.matmul ins(%m9, %W10 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e10 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %index = arith.constant 0 : index
  %result = tensor.extract %m10[%index, %index] : tensor<1024x1024xf32>
  return %result : f32
}
