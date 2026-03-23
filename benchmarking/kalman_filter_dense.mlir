func.func @kernel() -> f32{
    %A = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>
    %P0 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>
    %Q = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<1024x1024xf32>

    %e0 = arith.constant dense<0.0>: tensor<1024x1024xf32>
    %e1 = arith.constant dense<0.0>: tensor<1024x1024xf32>
    %e2 = arith.constant dense<0.0>: tensor<1024x1024xf32>
    %e3 = arith.constant dense<0.0>: tensor<1024x1024xf32>

    %At = linalg.transpose ins(%A  : tensor<1024x1024xf32>)
                           outs(%e0 : tensor<1024x1024xf32>)
                           permutation = [1, 0]

    %R1 = linalg.matmul ins(%A, %P0 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                        outs(%e1 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

    %R2 = linalg.matmul ins(%R1, %At: tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                        outs(%e2 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

    %P1 = linalg.add ins(%R2, %Q : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                     outs(%e3 : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

    %zero = arith.constant 0: index
    %result = tensor.extract %P1[%zero, %zero]: tensor<1024x1024xf32>
    return %result: f32
}
