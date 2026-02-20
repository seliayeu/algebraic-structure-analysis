// RUN: %build/tools/alg-opt %s --banded-analysis | FileCheck %s

module {
  func.func @test_matmul() {
    %out = tensor.empty() : tensor<3x3xf32>

    %diag = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1] } } 
            dense<[[2.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
    %id = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 0 : i64, propertyDims = [0, 1] } } 
          dense<[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]> : tensor<3x3xf32>
    %upper = arith.constant { metadata = { upperBw = 2 : i64, lowerBw = 0 : i64, propertyDims = [0, 1] } } 
             dense<[[2.0,4.0,5.0],[0.0,3.0,9.0],[0.0,0.0,7.0]]> : tensor<3x3xf32>
    %lower = arith.constant { metadata = { upperBw = 0 : i64, lowerBw = 2 : i64, propertyDims = [0, 1] } } 
             dense<[[2.0,0.0,0.0],[4.0,3.0,0.0],[5.0,9.0,7.0]]> : tensor<3x3xf32>
    %general = arith.constant { metadata = { upperBw = 2 : i64, lowerBw = 2 : i64, propertyDims = [0, 1] } } 
             dense<[[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]> : tensor<3x3xf32>
    %unmarked = arith.constant dense<[[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]]> : tensor<3x3xf32>

    // (0,0) * (0,0) -> (0,0)
    // CHECK: linalg.matmul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: upperBw = 0
    %res_diag_diag = linalg.matmul ins(%diag, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    
    // (0,0) * (0,0) -> (0,0)
    // CHECK: linalg.matmul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: upperBw = 0
    %res_diag_id = linalg.matmul ins(%diag, %id : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    
    // (0,0) * (2,0) -> (2,0)
    // CHECK: linalg.matmul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: upperBw = 2
    %res_diag_up = linalg.matmul ins(%diag, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    
    // (0,0) * (0,2) -> (0,2)
    // CHECK: linalg.matmul
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: upperBw = 0
    %res_diag_low = linalg.matmul ins(%diag, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (0,0) * (2,2) -> (2,2)
    // CHECK: linalg.matmul
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: upperBw = 2
    %res_diag_gen = linalg.matmul ins(%diag, %general : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (2,0) * (0,0) -> (2,0)
    // CHECK: linalg.matmul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: upperBw = 2
    %res_up_diag = linalg.matmul ins(%upper, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    
    // (2,0) * (2,0) -> (4,0) -> (Sat 2,0)
    // CHECK: linalg.matmul
    // CHECK-SAME: lowerBw = 0
    // CHECK-SAME: upperBw = 2
    %res_up_up = linalg.matmul ins(%upper, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    
    // (2,0) * (0,2) -> (2,2)
    // CHECK: linalg.matmul
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: upperBw = 2
    %res_up_low = linalg.matmul ins(%upper, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    
    // (2,0) * (2,2) -> (4,2) -> (2,2)
    // CHECK: linalg.matmul
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: upperBw = 2
    %res_up_gen = linalg.matmul ins(%upper, %general : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // (0,2) * (2,0) -> (2,2)
    // CHECK: linalg.matmul
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: upperBw = 2
    %res_low_up = linalg.matmul ins(%lower, %upper : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    
    // (0,2) * (0,2) -> (0,4) -> (0,2)
    // CHECK: linalg.matmul
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: upperBw = 0
    %res_low_low = linalg.matmul ins(%lower, %lower : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>
    
    // (0,2) * (2,2) -> (2,4) -> (2,2)
    // CHECK: linalg.matmul
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: upperBw = 2
    %res_low_gen = linalg.matmul ins(%lower, %general : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // unmarked * (0,0) -> (2,2)
    // CHECK: linalg.matmul
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: upperBw = 2
    %res_unmarked_diag = linalg.matmul ins(%unmarked, %diag : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    // unmarked * unmarked -> (2,2)
    // CHECK: linalg.matmul
    // CHECK-SAME: lowerBw = 2
    // CHECK-SAME: upperBw = 2
    %res_unmarked_unmarked = linalg.matmul ins(%unmarked, %unmarked : tensor<3x3xf32>, tensor<3x3xf32>) outs(%out : tensor<3x3xf32>) -> tensor<3x3xf32>

    return
  }
}
