func.func @kernel() -> f32 {
  %e1 = arith.constant dense<0.0> : tensor<4x2048x2048xf32>
  %e2 = arith.constant dense<0.0> : tensor<4x2048x2048xf32>
  %e3 = arith.constant dense<0.0> : tensor<4x2048x2048xf32>
  %e4 = arith.constant dense<0.0> : tensor<4x2048x2048xf32>
  %e5 = arith.constant dense<0.0> : tensor<4x2048x2048xf32>
  %e6 = arith.constant dense<0.0> : tensor<4x2048x2048xf32>
  %e7 = arith.constant dense<0.0> : tensor<4x2048x2048xf32>
  %e8 = arith.constant dense<0.0> : tensor<4x2048x2048xf32>
  %e9 = arith.constant dense<0.0> : tensor<4x2048x2048xf32>
  %e10 = arith.constant dense<0.0> : tensor<4x2048x2048xf32>

  %input = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<4x2048x2048xf32>
  %W1 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<4x2048x2048xf32>
  %W2 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<4x2048x2048xf32>
  %W3 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<4x2048x2048xf32>
  %W4 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<4x2048x2048xf32>
  %W5 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<4x2048x2048xf32>
  %W6 = arith.constant {metadata = {lowerBw = 1 : i64, upperBw = 1 : i64, propertyDims = [1, 2]}} dense<1.0> : tensor<4x2048x2048xf32>

  %m1 = linalg.batch_matmul ins(%input, %W1 : tensor<4x2048x2048xf32>, tensor<4x2048x2048xf32>) outs(%e1 : tensor<4x2048x2048xf32>) -> tensor<4x2048x2048xf32>
  %m2 = linalg.batch_matmul ins(%input, %W2 : tensor<4x2048x2048xf32>, tensor<4x2048x2048xf32>) outs(%e2 : tensor<4x2048x2048xf32>) -> tensor<4x2048x2048xf32>
  %m3 = linalg.batch_matmul ins(%input, %W3 : tensor<4x2048x2048xf32>, tensor<4x2048x2048xf32>) outs(%e3 : tensor<4x2048x2048xf32>) -> tensor<4x2048x2048xf32>

  %m4 = linalg.batch_matmul ins(%m2, %m3 : tensor<4x2048x2048xf32>, tensor<4x2048x2048xf32>) outs(%e4 : tensor<4x2048x2048xf32>) -> tensor<4x2048x2048xf32>
  %m5 = linalg.batch_matmul ins(%m4, %m1 : tensor<4x2048x2048xf32>, tensor<4x2048x2048xf32>) outs(%e5 : tensor<4x2048x2048xf32>) -> tensor<4x2048x2048xf32>
  %m6 = linalg.batch_matmul ins(%m5, %W4 : tensor<4x2048x2048xf32>, tensor<4x2048x2048xf32>) outs(%e6 : tensor<4x2048x2048xf32>) -> tensor<4x2048x2048xf32>

  %a1 = linalg.add ins(%m6, %input : tensor<4x2048x2048xf32>, tensor<4x2048x2048xf32>) outs(%e7 : tensor<4x2048x2048xf32>) -> tensor<4x2048x2048xf32>

  %m7 = linalg.batch_matmul ins(%a1, %W5 : tensor<4x2048x2048xf32>, tensor<4x2048x2048xf32>) outs(%e8 : tensor<4x2048x2048xf32>) -> tensor<4x2048x2048xf32>
  %m8 = linalg.batch_matmul ins(%m7, %W6 : tensor<4x2048x2048xf32>, tensor<4x2048x2048xf32>) outs(%e9 : tensor<4x2048x2048xf32>) -> tensor<4x2048x2048xf32>

  %a2 = linalg.add ins(%m8, %input : tensor<4x2048x2048xf32>, tensor<4x2048x2048xf32>) outs(%e10 : tensor<4x2048x2048xf32>) -> tensor<4x2048x2048xf32>

  %index = arith.constant 0: index
  %result = tensor.extract %a2[%index, %index, %index] : tensor<4x2048x2048xf32>

  return %result : f32
}

