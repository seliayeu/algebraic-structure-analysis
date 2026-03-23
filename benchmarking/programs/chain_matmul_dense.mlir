func.func @kernel() -> f32{
    %I0 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<5x5xf32>
    %I1 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<5x5xf32>
    %I2 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<5x5xf32>
    %I3 = arith.constant {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0,1]}} dense<1.0> : tensor<5x5xf32>

    %e0 = arith.constant dense<0.0>: tensor<5x5xf32>
    %e1 = arith.constant dense<0.0>: tensor<5x5xf32>
    %e2 = arith.constant dense<0.0>: tensor<5x5xf32>

    %R1 = linalg.matmul ins(%I0, %I1 : tensor<5x5xf32>, tensor<5x5xf32>)
                        outs(%e0 : tensor<5x5xf32>) -> tensor<5x5xf32>

    %R2 = linalg.matmul ins(%R1, %I2: tensor<5x5xf32>, tensor<5x5xf32>)
                        outs(%e1 : tensor<5x5xf32>) -> tensor<5x5xf32>

    %R3 = linalg.matmul ins(%R2, %I3: tensor<5x5xf32>, tensor<5x5xf32>)
                        outs(%e2 : tensor<5x5xf32>) -> tensor<5x5xf32>

    %zero = arith.constant 0: index
    %result = tensor.extract %R3[%zero, %zero]: tensor<5x5xf32>
    return %result: f32
}
