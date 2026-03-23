func.func @kernel() -> f32 {

  %0 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>
  %1 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>

  %2 = arith.constant dense<0.0>: tensor<1024x1024xf32>
  %3 = linalg.matmul ins(%0, %1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
        outs(%2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>


  %4 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>
  %5 = arith.constant dense<0.0>: tensor<1024x1024xf32>
  %6 = linalg.matmul ins(%3, %4 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
        outs(%5 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>


  %7 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>
  %8 = arith.constant dense<0.0>: tensor<1024x1024xf32>
  %9 = linalg.matmul ins(%6, %7 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
        outs(%8 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>


  %10 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>
  %11 = arith.constant dense<0.0>: tensor<1024x1024xf32>
  %12 = linalg.matmul ins(%9, %10 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
        outs(%11 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>


  %13 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>
  %14 = arith.constant dense<0.0>: tensor<1024x1024xf32>
  %15 = linalg.add ins(%12, %13 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
        outs(%14 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>


  %16 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>
  %17 = arith.constant dense<0.0>: tensor<1024x1024xf32>
  %18 = linalg.matmul ins(%15, %16 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
        outs(%17 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %19 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>
  %20 = arith.constant dense<0.0>: tensor<1024x1024xf32>
  %21 = linalg.matmul ins(%18, %19 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
        outs(%20 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>


  %22 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>
  %23 = arith.constant dense<0.0>: tensor<1024x1024xf32>
  %24 = linalg.mul ins(%21, %22 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
        outs(%23 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>


  %25 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>
  %26 = arith.constant dense<0.0>: tensor<1024x1024xf32>
  %27 = linalg.add ins(%24, %25 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
        outs(%23 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>


  %28 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>
  %29 = arith.constant dense<0.0>: tensor<1024x1024xf32>
  %30 = linalg.add ins(%27, %28 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
        outs(%29 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %zero = arith.constant 0: index
  %result = tensor.extract %30[%zero, %zero]: tensor<1024x1024xf32>
  return %result: f32
}
