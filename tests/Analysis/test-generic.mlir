// RUN: %build/tools/alg-opt %s \
// RUN:  --algebraic-structure-debug

module {
  func.func @test_generic_non_contiguous_dims() {
    %out = tensor.empty() : tensor<2x2x2xf32>

    // ---------------------------------------------------------
    // LHS: Rank 4 Tensor (Shape: 2x1x2x2)
    // Structure: Defined on dims [0, 3] ('a' and 'd')
    // This effectively treats (b, c) as batch dims between the matrix rows/cols.
    // ---------------------------------------------------------

    %lhs_split = arith.constant { 
      metadata = { 
        analysisState = "Symmetric", 
        propertyDims = [0, 3]   // <--- Non-contiguous!
      } 
    } dense<2.0> : tensor<2x1x2x2xf32>

    // ---------------------------------------------------------
    // RHS: Rank 3 Tensor (Shape: 2x2x1)
    // Structure: Defined on dims [0, 1] ('d' and 'f')
    // ---------------------------------------------------------

    %rhs_std = arith.constant { 
      metadata = { 
        analysisState = "Identity", 
        propertyDims = [0, 1] 
      } 
    } dense<2.0> : tensor<2x2x1xf32>


    // ---------------------------------------------------------
    // Test 1: Contraction 'abcd, dfg -> afc'
    //
    // Iteration Space: (a, b, c, d, f, g)
    // LHS Map: (a, b, c, d)  -> properties on 'a'(d0) and 'd'(d3)
    // RHS Map: (d, f, g)     -> properties on 'd'(d3) and 'f'(d4)
    // Res Map: (a, f, c)     -> Result indices: 0='a', 1='f', 2='c'
    //
    // Analysis Logic:
    // 1. Match K: LHS col (d3) == RHS row (d3). Match!
    // 2. Find M: LHS row (d0). Found in Result at index 0.
    // 3. Find N: RHS col (d4). Found in Result at index 1.
    // 4. Result propertyDims should be [0, 1].
    // ---------------------------------------------------------

    // Symmetric (on 0,3) * Identity (on 0,1) -> Symmetric (on 0,1)
    // CHECK: analysisState = "Symmetric"
    // CHECK-SAME: propertyDims = [0, 1]
    %res_split = linalg.generic {
      indexing_maps = [
        // LHS: (a, b, c, d)
        affine_map<(a, b, c, d, f, g) -> (a, b, c, d)>, 
        // RHS: (d, f, g)
        affine_map<(a, b, c, d, f, g) -> (d, f, g)>, 
        // Res: (a, f, c)
        affine_map<(a, b, c, d, f, g) -> (a, f, c)>
      ],
      iterator_types = ["parallel", "reduction", "parallel", "reduction", "parallel", "reduction"]
    } ins(%lhs_split, %rhs_std : tensor<2x1x2x2xf32>, tensor<2x2x1xf32>) outs(%out : tensor<2x2x2xf32>) {
    ^bb0(%in_lhs: f32, %in_rhs: f32, %acc: f32):
      %0 = arith.mulf %in_lhs, %in_rhs : f32
      %1 = arith.addf %acc, %0 : f32
      linalg.yield %1 : f32
    } -> tensor<2x2x2xf32>


    // ---------------------------------------------------------
    // Test 2: Permuted Result 'abcd, dfg -> cfa'
    //
    // Same input structures.
    // Res Map: (c, f, a)     -> Result indices: 0='c', 1='f', 2='a'
    //
    // Analysis Logic:
    // 1. Match K: LHS col (d3) == RHS row (d3). Match!
    // 2. Find M: LHS row (d0). Found in Result at index 2 ('a').
    // 3. Find N: RHS col (d4). Found in Result at index 1 ('f').
    // 4. Result propertyDims should be [2, 1] (Transposed structure in output).
    // ---------------------------------------------------------

    // Symmetric * Identity -> Symmetric (on 2,1)
    // CHECK: analysisState = "Symmetric"
    // CHECK-SAME: propertyDims = [2, 1]
    %res_permuted = linalg.generic {
      indexing_maps = [
        // LHS: (a, b, c, d)
        affine_map<(a, b, c, d, f, g) -> (a, b, c, d)>, 
        // RHS: (d, f, g)
        affine_map<(a, b, c, d, f, g) -> (d, f, g)>, 
        // Res: (c, f, a)
        affine_map<(a, b, c, d, f, g) -> (c, f, a)>
      ],
      iterator_types = ["parallel", "reduction", "parallel", "reduction", "parallel", "reduction"]
    } ins(%lhs_split, %rhs_std : tensor<2x1x2x2xf32>, tensor<2x2x1xf32>) outs(%out : tensor<2x2x2xf32>) {
    ^bb0(%in_lhs: f32, %in_rhs: f32, %acc: f32):
      %0 = arith.mulf %in_lhs, %in_rhs : f32
      %1 = arith.addf %acc, %0 : f32
      linalg.yield %1 : f32
    } -> tensor<2x2x2xf32>

    return
  }
}
