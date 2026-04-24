// ============================================================
// Naive Sparse Attention in MLIR — Static Shapes (1024x1024)
//
// Dialects: func, linalg, arith, math, tensor, scf
// ============================================================

// ============================================================
// 1. Scaled QK^T   [1024 x 1024]
// ============================================================
func.func @scaled_qkt(
    %Q : tensor<1024x1024xf32>,
    %K : tensor<1024x1024xf32>
) -> tensor<1024x1024xf32> {

  // scale = 1 / sqrt(1024) = 1 / 32 = 0.03125
  %scale = arith.constant 0.03125 : f32
  %zero  = arith.constant 0.0 : f32
  
  // ---- Transpose K ----
  %init_kT = tensor.empty() : tensor<1024x1024xf32>
  %kT = linalg.transpose 
    ins(%K : tensor<1024x1024xf32>)
    outs(%init_kT : tensor<1024x1024xf32>) permutation = [1, 0]

  // ---- Matmul Q @ K^T ----
  %init_qkt  = tensor.empty() : tensor<1024x1024xf32>
  %init_qktZ = linalg.fill ins(%zero : f32)
                           outs(%init_qkt : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %qkt = linalg.matmul 
    ins(%Q, %kT : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
    outs(%init_qktZ : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  // ---- Multiply every element by the scale factor ----
  %init_out = tensor.empty() : tensor<1024x1024xf32>
  %out = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,
      affine_map<(i, j) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%qkt : tensor<1024x1024xf32>)
    outs(%init_out : tensor<1024x1024xf32>) {
    ^bb0(%x : f32, %_ : f32):
      %v = arith.mulf %x, %scale : f32
      linalg.yield %v : f32
  } -> tensor<1024x1024xf32>

  return %out : tensor<1024x1024xf32>
}

// ============================================================
// 2. Apply dense boolean mask 
// ============================================================
func.func @apply_mask(
    %scores : tensor<1024x1024xf32>,
    %mask   : tensor<1024x1024xi1>
) -> tensor<1024x1024xf32> {

  // -inf as an f32 bit pattern
  %neg_inf_i = arith.constant 0xFF800000 : i32
  %neg_inf   = arith.bitcast %neg_inf_i  : i32 to f32

  %init = tensor.empty() : tensor<1024x1024xf32>

  %out = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,   // scores
      affine_map<(i, j) -> (i, j)>,   // mask
      affine_map<(i, j) -> (i, j)>    // out
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%scores, %mask : tensor<1024x1024xf32>, tensor<1024x1024xi1>)
    outs(%init : tensor<1024x1024xf32>) {
    ^bb0(%s : f32, %m : i1, %_ : f32):
      %v = arith.select %m, %s, %neg_inf : f32
      linalg.yield %v : f32
  } -> tensor<1024x1024xf32>

  return %out : tensor<1024x1024xf32>
}

// ============================================================
// 3. Numerically stable row-wise softmax
// ============================================================
func.func @row_softmax(
    %x : tensor<1024x1024xf32>
) -> tensor<1024x1024xf32> {

  %zero_f    = arith.constant 0.0        : f32
  %neg_inf_i = arith.constant 0xFF800000 : i32
  %neg_inf   = arith.bitcast %neg_inf_i  : i32 to f32

  // ---- (a) row max ----
  %max_buf  = tensor.empty() : tensor<1024xf32>
  %max_init = linalg.fill ins(%neg_inf : f32)
                          outs(%max_buf : tensor<1024xf32>) -> tensor<1024xf32>

  %row_max = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,
      affine_map<(i, j) -> (i)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%x : tensor<1024x1024xf32>)
    outs(%max_init : tensor<1024xf32>) {
    ^bb0(%v : f32, %m : f32):
      %nm = arith.maximumf %v, %m : f32
      linalg.yield %nm : f32
  } -> tensor<1024xf32>

  // ---- (b) exp(x - row_max) ----
  %exp_buf  = tensor.empty() : tensor<1024x1024xf32>
  %exp_init = linalg.fill ins(%zero_f : f32)
                          outs(%exp_buf : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %exp_out = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,
      affine_map<(i, j) -> (i)>,
      affine_map<(i, j) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%x, %row_max : tensor<1024x1024xf32>, tensor<1024xf32>)
    outs(%exp_init : tensor<1024x1024xf32>) {
    ^bb0(%v : f32, %m : f32, %_ : f32):
      %sh = arith.subf %v, %m  : f32
      %e  = math.exp  %sh      : f32
      linalg.yield %e : f32
  } -> tensor<1024x1024xf32>

  // ---- (c) row sum of exp ----
  %sum_buf  = tensor.empty() : tensor<1024xf32>
  %sum_init = linalg.fill ins(%zero_f : f32)
                          outs(%sum_buf : tensor<1024xf32>) -> tensor<1024xf32>

  %row_sum = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,
      affine_map<(i, j) -> (i)>
    ],
    iterator_types = ["parallel", "reduction"]
  } ins(%exp_out : tensor<1024x1024xf32>)
    outs(%sum_init : tensor<1024xf32>) {
    ^bb0(%e : f32, %s : f32):
      %ns = arith.addf %s, %e : f32
      linalg.yield %ns : f32
  } -> tensor<1024xf32>

  // ---- (d) normalise ----
  %norm_buf  = tensor.empty() : tensor<1024x1024xf32>
  %norm_init = linalg.fill ins(%zero_f : f32)
                             outs(%norm_buf : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %softmax = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,
      affine_map<(i, j) -> (i)>,
      affine_map<(i, j) -> (i, j)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%exp_out, %row_sum : tensor<1024x1024xf32>, tensor<1024xf32>)
    outs(%norm_init : tensor<1024x1024xf32>) {
    ^bb0(%e : f32, %s : f32, %_ : f32):
      %n = arith.divf %e, %s : f32
      linalg.yield %n : f32
  } -> tensor<1024x1024xf32>

  return %softmax : tensor<1024x1024xf32>
}

// ============================================================
// 4. Weighted sum over V:  attn_weights @ V
// ============================================================
func.func @attn_output(
    %w : tensor<1024x1024xf32>,
    %V : tensor<1024x1024xf32>
) -> tensor<1024x1024xf32> {

  %zero  = arith.constant 0.0 : f32
  %init  = tensor.empty() : tensor<1024x1024xf32>
  %initZ = linalg.fill ins(%zero : f32)
                       outs(%init : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  // Weights @ V using standard linalg.matmul
  %out = linalg.matmul 
    ins(%w, %V : tensor<1024x1024xf32>, tensor<1024x1024xf32>)
    outs(%initZ : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  return %out : tensor<1024x1024xf32>
}

// ============================================================
// 5. Top-level entry point
// ============================================================
func.func @sparse_attention(
    %Q    : tensor<1024x1024xf32>,
    %K    : tensor<1024x1024xf32>,
    %V    : tensor<1024x1024xf32>,
    %mask : tensor<1024x1024xi1>
) -> tensor<1024x1024xf32> {

  %scores  = func.call @scaled_qkt(%Q, %K)
             : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %masked  = func.call @apply_mask(%scores, %mask)
             : (tensor<1024x1024xf32>, tensor<1024x1024xi1>) -> tensor<1024x1024xf32>

  %weights = func.call @row_softmax(%masked)
             : (tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  %out     = func.call @attn_output(%weights, %V)
             : (tensor<1024x1024xf32>, tensor<1024x1024xf32>) -> tensor<1024x1024xf32>

  return %out : tensor<1024x1024xf32>
}

// ============================================================
// main (Simplified for Static Verification)
// ============================================================
func.func @kernel() -> f32 {
  %Q    = arith.constant dense<1.0> : tensor<1024x1024xf32>
  %K    = arith.constant dense<1.0> : tensor<1024x1024xf32>
  %V    = arith.constant dense<1.0> : tensor<1024x1024xf32>
  %mask = arith.constant dense<true> : tensor<1024x1024xi1>

  %out = func.call @sparse_attention(%Q, %K, %V, %mask)
         : (tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xf32>, tensor<1024x1024xi1>)
           -> tensor<1024x1024xf32>

  %zidx = arith.constant 0: index
  %result = tensor.extract %out[%zidx, %zidx]: tensor<1024x1024xf32>

  return %result : f32
}
