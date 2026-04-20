func.func @main () -> tensor<10x10xf32> {
    %0 = arith.constant {metadata={lowerBw = 5 : i64, upperBw = 0 : i64, propertyDims=[0, 1]}} dense<1.0> : tensor<10x10xf32>
    %2 = dia.softmax (%0: tensor<10x10xf32>) -> tensor<10x10xf32>
    return %2: tensor<10x10xf32>
  }
