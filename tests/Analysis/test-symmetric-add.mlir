// RUN: %build/tools/alg-opt %s \
// RUN:  --algebraic-structure-debug

module {
    func.func @main() {
        %0 = arith.constant { metadata = { analysisState = "Symmetric", propertyDims = [0, 1] } } dense<[[2.0,4.0,5.0],[4.0,3.0,9.0],[5.0,9.0,7.0]]> : tensor<3x3xf32>
        %1 = arith.constant { metadata = { analysisState = "Symmetric", propertyDims = [0, 1] } } dense<[[5.0,8.0,2.0],[8.0,4.0,3.0],[2.0,3.0,3.0]]> : tensor<3x3xf32>
        %3 = tensor.empty() : tensor<3x3xf32>
        
        // CHECK: analysisState = "Symmetric"
        // CHECK-SAME: propertyDims = [0, 1]
        %2 = linalg.add
            ins(%0, %1 : tensor<3x3xf32>, tensor<3x3xf32>)
            outs(%3 : tensor<3x3xf32>)
            -> tensor<3x3xf32>

        return
    }
}
