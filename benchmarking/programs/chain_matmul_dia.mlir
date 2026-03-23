func.func @kernel() -> f32 {
    %I0 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} dense<1.0> : tensor<1024x1024xf32>
    %I1 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} dense<1.0> : tensor<1024x1024xf32>
    %I2 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} dense<1.0> : tensor<1024x1024xf32>
    %I3 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} dense<1.0> : tensor<1024x1024xf32>

    %c0 = arith.constant 0 : index
    %e0 = tensor.empty(%c0) : tensor<?x1024xf32>
    %e1 = tensor.empty(%c0) : tensor<?x1024xf32>
    %e2 = tensor.empty(%c0) : tensor<?x1024xf32>

    %R1 = dia.matmul ins(%I0, %I1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
                      outs(%e0 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
    %R2 = dia.matmul ins(%R1, %I2 : tensor<?x1024xf32>, tensor<1024x1024xf32>)
                      outs(%e1 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
    %R3 = dia.matmul ins(%R2, %I3 : tensor<?x1024xf32>, tensor<1024x1024xf32>)
                      outs(%e2 : tensor<?x1024xf32>) -> tensor<?x1024xf32>

    %index = arith.constant 5: index
    %result = tensor.extract %R3[%index, %index] : tensor<?x1024xf32>
    return %result : f32
}
