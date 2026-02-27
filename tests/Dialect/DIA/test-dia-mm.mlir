// test-dia-matmul.mlir
module {
  func.func @main() -> tensor<3x4xf32> {
    %tridiag = arith.constant {metadata = {layout = "dia", upperBw = 1 : i64, lowerBw = 1 : i64, propertyDims = [0, 1]}}
        dense<[[0.0, 8.0, 9.0, 1.0],
               [1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 0.0]]> : tensor<3x4xf32>
    %diag = arith.constant {metadata = {layout = "dia", upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1]}}
        dense<[[1.0, 2.0, 3.0, 4.0]]> : tensor<1x4xf32>
    %0 = tensor.empty() : tensor<3x4xf32>
    %result = dia.matmul ins(%tridiag, %diag : tensor<3x4xf32>, tensor<1x4xf32>)
                         outs(%0 : tensor<3x4xf32>)
                         -> tensor<3x4xf32>
    return %result : tensor<3x4xf32>
  }
}
