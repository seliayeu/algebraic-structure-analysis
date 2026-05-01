func.func @kernel() -> f32 {
  %e1 = tensor.empty() : tensor<1024x1024xf32>
  %e2 = tensor.empty() : tensor<1024x1024xf32>
  %e3 = tensor.empty() : tensor<1024x1024xf32>
  %e4 = tensor.empty() : tensor<1024x1024xf32>
  %e5 = tensor.empty() : tensor<1024x1024xf32>
  %e6 = tensor.empty() : tensor<1024x1024xf32>
  %e7 = tensor.empty() : tensor<1024x1024xf32>
  %e8 = tensor.empty() : tensor<1024x1024xf32>
  %e9 = tensor.empty() : tensor<1024x1024xf32>
  %e10 = tensor.empty() : tensor<1024x1024xf32>

  %input = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} dense<1.0> : tensor<1024x1024xf32>
  %W1 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} dense<1.0> : tensor<1024x1024xf32>
  %W2 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} dense<1.0> : tensor<1024x1024xf32>
  %W3 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} dense<1.0> : tensor<1024x1024xf32>
  %W4 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} dense<1.0> : tensor<1024x1024xf32>
  %W5 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} dense<1.0> : tensor<1024x1024xf32>
  %W6 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} dense<1.0> : tensor<1024x1024xf32>

  %m1 = linalg.matmul ins(%input, %W1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %m2 = linalg.matmul ins(%input, %W2 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %m3 = linalg.matmul ins(%input, %W3 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e3 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %m4 = linalg.matmul ins(%m2, %m3 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e4 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %m5 = linalg.matmul ins(%m4, %m1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e5 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %m6 = linalg.matmul ins(%m5, %W4 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e6 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %a1 = linalg.add ins(%m6, %input : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e7 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %m7 = linalg.matmul ins(%a1, %W5 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e8 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %m8 = linalg.matmul ins(%m7, %W6 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e9 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %a2 = linalg.add ins(%m8, %input : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e10 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %index = arith.constant 5: index
  %result = tensor.extract %a2[%index, %index] : tensor<1024x1024xf32>
  return %result : f32
}

