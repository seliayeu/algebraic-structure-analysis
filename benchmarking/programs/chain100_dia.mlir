func.func @kernel() -> f32 {
  %c0 = arith.constant 0 : index
  %cf1 = arith.constant 1.0 : f32
  %e1 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e2 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e3 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e4 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e5 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e6 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e7 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e8 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e9 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e10 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e11 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e12 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e13 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e14 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e15 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e16 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e17 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e18 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e19 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e20 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e21 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e22 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e23 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e24 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e25 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e26 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e27 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e28 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e29 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e30 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e31 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e32 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e33 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e34 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e35 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e36 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e37 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e38 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e39 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e40 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e41 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e42 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e43 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e44 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e45 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e46 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e47 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e48 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e49 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e50 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e51 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e52 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e53 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e54 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e55 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e56 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e57 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e58 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e59 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e60 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e61 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e62 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e63 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e64 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e65 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e66 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e67 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e68 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e69 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e70 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e71 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e72 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e73 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e74 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e75 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e76 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e77 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e78 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e79 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e80 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e81 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e82 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e83 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e84 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e85 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e86 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e87 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e88 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e89 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e90 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e91 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e92 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e93 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e94 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e95 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e96 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e97 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e98 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e99 = tensor.empty(%c0) : tensor<?x1024xf32>
  %e100 = tensor.empty(%c0) : tensor<?x1024xf32>
  %input_empty = tensor.empty() : tensor<1024x1024xf32>
  %W1_empty = tensor.empty() : tensor<1024x1024xf32>
  %W2_empty = tensor.empty() : tensor<1024x1024xf32>
  %W3_empty = tensor.empty() : tensor<1024x1024xf32>
  %W4_empty = tensor.empty() : tensor<1024x1024xf32>
  %W5_empty = tensor.empty() : tensor<1024x1024xf32>
  %W6_empty = tensor.empty() : tensor<1024x1024xf32>
  %W7_empty = tensor.empty() : tensor<1024x1024xf32>
  %W8_empty = tensor.empty() : tensor<1024x1024xf32>
  %W9_empty = tensor.empty() : tensor<1024x1024xf32>
  %W10_empty = tensor.empty() : tensor<1024x1024xf32>
  %W11_empty = tensor.empty() : tensor<1024x1024xf32>
  %W12_empty = tensor.empty() : tensor<1024x1024xf32>
  %W13_empty = tensor.empty() : tensor<1024x1024xf32>
  %W14_empty = tensor.empty() : tensor<1024x1024xf32>
  %W15_empty = tensor.empty() : tensor<1024x1024xf32>
  %W16_empty = tensor.empty() : tensor<1024x1024xf32>
  %W17_empty = tensor.empty() : tensor<1024x1024xf32>
  %W18_empty = tensor.empty() : tensor<1024x1024xf32>
  %W19_empty = tensor.empty() : tensor<1024x1024xf32>
  %W20_empty = tensor.empty() : tensor<1024x1024xf32>
  %W21_empty = tensor.empty() : tensor<1024x1024xf32>
  %W22_empty = tensor.empty() : tensor<1024x1024xf32>
  %W23_empty = tensor.empty() : tensor<1024x1024xf32>
  %W24_empty = tensor.empty() : tensor<1024x1024xf32>
  %W25_empty = tensor.empty() : tensor<1024x1024xf32>
  %W26_empty = tensor.empty() : tensor<1024x1024xf32>
  %W27_empty = tensor.empty() : tensor<1024x1024xf32>
  %W28_empty = tensor.empty() : tensor<1024x1024xf32>
  %W29_empty = tensor.empty() : tensor<1024x1024xf32>
  %W30_empty = tensor.empty() : tensor<1024x1024xf32>
  %W31_empty = tensor.empty() : tensor<1024x1024xf32>
  %W32_empty = tensor.empty() : tensor<1024x1024xf32>
  %W33_empty = tensor.empty() : tensor<1024x1024xf32>
  %W34_empty = tensor.empty() : tensor<1024x1024xf32>
  %W35_empty = tensor.empty() : tensor<1024x1024xf32>
  %W36_empty = tensor.empty() : tensor<1024x1024xf32>
  %W37_empty = tensor.empty() : tensor<1024x1024xf32>
  %W38_empty = tensor.empty() : tensor<1024x1024xf32>
  %W39_empty = tensor.empty() : tensor<1024x1024xf32>
  %W40_empty = tensor.empty() : tensor<1024x1024xf32>
  %W41_empty = tensor.empty() : tensor<1024x1024xf32>
  %W42_empty = tensor.empty() : tensor<1024x1024xf32>
  %W43_empty = tensor.empty() : tensor<1024x1024xf32>
  %W44_empty = tensor.empty() : tensor<1024x1024xf32>
  %W45_empty = tensor.empty() : tensor<1024x1024xf32>
  %W46_empty = tensor.empty() : tensor<1024x1024xf32>
  %W47_empty = tensor.empty() : tensor<1024x1024xf32>
  %W48_empty = tensor.empty() : tensor<1024x1024xf32>
  %W49_empty = tensor.empty() : tensor<1024x1024xf32>
  %W50_empty = tensor.empty() : tensor<1024x1024xf32>
  %W51_empty = tensor.empty() : tensor<1024x1024xf32>
  %W52_empty = tensor.empty() : tensor<1024x1024xf32>
  %W53_empty = tensor.empty() : tensor<1024x1024xf32>
  %W54_empty = tensor.empty() : tensor<1024x1024xf32>
  %W55_empty = tensor.empty() : tensor<1024x1024xf32>
  %W56_empty = tensor.empty() : tensor<1024x1024xf32>
  %W57_empty = tensor.empty() : tensor<1024x1024xf32>
  %W58_empty = tensor.empty() : tensor<1024x1024xf32>
  %W59_empty = tensor.empty() : tensor<1024x1024xf32>
  %W60_empty = tensor.empty() : tensor<1024x1024xf32>
  %W61_empty = tensor.empty() : tensor<1024x1024xf32>
  %W62_empty = tensor.empty() : tensor<1024x1024xf32>
  %W63_empty = tensor.empty() : tensor<1024x1024xf32>
  %W64_empty = tensor.empty() : tensor<1024x1024xf32>
  %W65_empty = tensor.empty() : tensor<1024x1024xf32>
  %W66_empty = tensor.empty() : tensor<1024x1024xf32>
  %W67_empty = tensor.empty() : tensor<1024x1024xf32>
  %W68_empty = tensor.empty() : tensor<1024x1024xf32>
  %W69_empty = tensor.empty() : tensor<1024x1024xf32>
  %W70_empty = tensor.empty() : tensor<1024x1024xf32>
  %W71_empty = tensor.empty() : tensor<1024x1024xf32>
  %W72_empty = tensor.empty() : tensor<1024x1024xf32>
  %W73_empty = tensor.empty() : tensor<1024x1024xf32>
  %W74_empty = tensor.empty() : tensor<1024x1024xf32>
  %W75_empty = tensor.empty() : tensor<1024x1024xf32>
  %W76_empty = tensor.empty() : tensor<1024x1024xf32>
  %W77_empty = tensor.empty() : tensor<1024x1024xf32>
  %W78_empty = tensor.empty() : tensor<1024x1024xf32>
  %W79_empty = tensor.empty() : tensor<1024x1024xf32>
  %W80_empty = tensor.empty() : tensor<1024x1024xf32>
  %W81_empty = tensor.empty() : tensor<1024x1024xf32>
  %W82_empty = tensor.empty() : tensor<1024x1024xf32>
  %W83_empty = tensor.empty() : tensor<1024x1024xf32>
  %W84_empty = tensor.empty() : tensor<1024x1024xf32>
  %W85_empty = tensor.empty() : tensor<1024x1024xf32>
  %W86_empty = tensor.empty() : tensor<1024x1024xf32>
  %W87_empty = tensor.empty() : tensor<1024x1024xf32>
  %W88_empty = tensor.empty() : tensor<1024x1024xf32>
  %W89_empty = tensor.empty() : tensor<1024x1024xf32>
  %W90_empty = tensor.empty() : tensor<1024x1024xf32>
  %W91_empty = tensor.empty() : tensor<1024x1024xf32>
  %W92_empty = tensor.empty() : tensor<1024x1024xf32>
  %W93_empty = tensor.empty() : tensor<1024x1024xf32>
  %W94_empty = tensor.empty() : tensor<1024x1024xf32>
  %W95_empty = tensor.empty() : tensor<1024x1024xf32>
  %W96_empty = tensor.empty() : tensor<1024x1024xf32>
  %W97_empty = tensor.empty() : tensor<1024x1024xf32>
  %W98_empty = tensor.empty() : tensor<1024x1024xf32>
  %W99_empty = tensor.empty() : tensor<1024x1024xf32>
  %W100_empty = tensor.empty() : tensor<1024x1024xf32>
  %input = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%input_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W1 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W1_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W2 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W2_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W3 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W3_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W4 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W4_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W5 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W5_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W6 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W6_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W7 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W7_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W8 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W8_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W9 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W9_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W10 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W10_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W11 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W11_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W12 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W12_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W13 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W13_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W14 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W14_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W15 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W15_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W16 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W16_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W17 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W17_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W18 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W18_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W19 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W19_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W20 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W20_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W21 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W21_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W22 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W22_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W23 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W23_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W24 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W24_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W25 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W25_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W26 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W26_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W27 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W27_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W28 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W28_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W29 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W29_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W30 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W30_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W31 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W31_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W32 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W32_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W33 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W33_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W34 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W34_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W35 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W35_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W36 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W36_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W37 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W37_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W38 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W38_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W39 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W39_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W40 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W40_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W41 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W41_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W42 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W42_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W43 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W43_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W44 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W44_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W45 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W45_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W46 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W46_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W47 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W47_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W48 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W48_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W49 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W49_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W50 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W50_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W51 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W51_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W52 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W52_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W53 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W53_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W54 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W54_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W55 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W55_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W56 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W56_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W57 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W57_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W58 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W58_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W59 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W59_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W60 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W60_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W61 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W61_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W62 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W62_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W63 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W63_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W64 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W64_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W65 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W65_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W66 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W66_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W67 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W67_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W68 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W68_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W69 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W69_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W70 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W70_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W71 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W71_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W72 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W72_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W73 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W73_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W74 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W74_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W75 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W75_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W76 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W76_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W77 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W77_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W78 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W78_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W79 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W79_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W80 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W80_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W81 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W81_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W82 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W82_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W83 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W83_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W84 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W84_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W85 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W85_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W86 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W86_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W87 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W87_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W88 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W88_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W89 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W89_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W90 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W90_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W91 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W91_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W92 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W92_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W93 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W93_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W94 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W94_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W95 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W95_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W96 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W96_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W97 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W97_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W98 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W98_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W99 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W99_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %W100 = linalg.fill {metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}} ins(%cf1 : f32) outs(%W100_empty : tensor<1024x1024xf32>) -> tensor<1024x1024xf32>
  %m1 = dia.matmul ins(%input, %W1 : tensor<1024x1024xf32>, tensor<1024x1024xf32>) outs(%e1 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m2 = dia.matmul ins(%m1, %W2 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e2 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m3 = dia.matmul ins(%m2, %W3 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e3 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m4 = dia.matmul ins(%m3, %W4 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e4 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m5 = dia.matmul ins(%m4, %W5 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e5 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m6 = dia.matmul ins(%m5, %W6 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e6 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m7 = dia.matmul ins(%m6, %W7 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e7 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m8 = dia.matmul ins(%m7, %W8 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e8 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m9 = dia.matmul ins(%m8, %W9 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e9 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m10 = dia.matmul ins(%m9, %W10 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e10 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m11 = dia.matmul ins(%m10, %W11 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e11 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m12 = dia.matmul ins(%m11, %W12 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e12 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m13 = dia.matmul ins(%m12, %W13 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e13 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m14 = dia.matmul ins(%m13, %W14 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e14 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m15 = dia.matmul ins(%m14, %W15 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e15 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m16 = dia.matmul ins(%m15, %W16 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e16 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m17 = dia.matmul ins(%m16, %W17 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e17 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m18 = dia.matmul ins(%m17, %W18 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e18 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m19 = dia.matmul ins(%m18, %W19 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e19 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m20 = dia.matmul ins(%m19, %W20 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e20 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m21 = dia.matmul ins(%m20, %W21 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e21 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m22 = dia.matmul ins(%m21, %W22 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e22 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m23 = dia.matmul ins(%m22, %W23 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e23 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m24 = dia.matmul ins(%m23, %W24 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e24 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m25 = dia.matmul ins(%m24, %W25 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e25 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m26 = dia.matmul ins(%m25, %W26 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e26 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m27 = dia.matmul ins(%m26, %W27 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e27 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m28 = dia.matmul ins(%m27, %W28 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e28 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m29 = dia.matmul ins(%m28, %W29 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e29 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m30 = dia.matmul ins(%m29, %W30 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e30 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m31 = dia.matmul ins(%m30, %W31 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e31 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m32 = dia.matmul ins(%m31, %W32 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e32 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m33 = dia.matmul ins(%m32, %W33 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e33 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m34 = dia.matmul ins(%m33, %W34 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e34 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m35 = dia.matmul ins(%m34, %W35 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e35 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m36 = dia.matmul ins(%m35, %W36 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e36 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m37 = dia.matmul ins(%m36, %W37 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e37 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m38 = dia.matmul ins(%m37, %W38 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e38 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m39 = dia.matmul ins(%m38, %W39 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e39 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m40 = dia.matmul ins(%m39, %W40 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e40 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m41 = dia.matmul ins(%m40, %W41 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e41 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m42 = dia.matmul ins(%m41, %W42 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e42 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m43 = dia.matmul ins(%m42, %W43 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e43 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m44 = dia.matmul ins(%m43, %W44 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e44 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m45 = dia.matmul ins(%m44, %W45 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e45 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m46 = dia.matmul ins(%m45, %W46 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e46 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m47 = dia.matmul ins(%m46, %W47 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e47 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m48 = dia.matmul ins(%m47, %W48 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e48 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m49 = dia.matmul ins(%m48, %W49 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e49 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m50 = dia.matmul ins(%m49, %W50 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e50 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m51 = dia.matmul ins(%m50, %W51 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e51 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m52 = dia.matmul ins(%m51, %W52 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e52 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m53 = dia.matmul ins(%m52, %W53 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e53 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m54 = dia.matmul ins(%m53, %W54 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e54 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m55 = dia.matmul ins(%m54, %W55 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e55 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m56 = dia.matmul ins(%m55, %W56 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e56 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m57 = dia.matmul ins(%m56, %W57 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e57 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m58 = dia.matmul ins(%m57, %W58 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e58 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m59 = dia.matmul ins(%m58, %W59 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e59 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m60 = dia.matmul ins(%m59, %W60 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e60 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m61 = dia.matmul ins(%m60, %W61 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e61 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m62 = dia.matmul ins(%m61, %W62 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e62 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m63 = dia.matmul ins(%m62, %W63 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e63 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m64 = dia.matmul ins(%m63, %W64 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e64 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m65 = dia.matmul ins(%m64, %W65 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e65 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m66 = dia.matmul ins(%m65, %W66 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e66 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m67 = dia.matmul ins(%m66, %W67 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e67 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m68 = dia.matmul ins(%m67, %W68 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e68 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m69 = dia.matmul ins(%m68, %W69 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e69 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m70 = dia.matmul ins(%m69, %W70 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e70 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m71 = dia.matmul ins(%m70, %W71 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e71 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m72 = dia.matmul ins(%m71, %W72 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e72 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m73 = dia.matmul ins(%m72, %W73 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e73 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m74 = dia.matmul ins(%m73, %W74 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e74 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m75 = dia.matmul ins(%m74, %W75 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e75 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m76 = dia.matmul ins(%m75, %W76 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e76 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m77 = dia.matmul ins(%m76, %W77 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e77 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m78 = dia.matmul ins(%m77, %W78 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e78 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m79 = dia.matmul ins(%m78, %W79 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e79 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m80 = dia.matmul ins(%m79, %W80 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e80 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m81 = dia.matmul ins(%m80, %W81 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e81 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m82 = dia.matmul ins(%m81, %W82 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e82 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m83 = dia.matmul ins(%m82, %W83 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e83 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m84 = dia.matmul ins(%m83, %W84 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e84 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m85 = dia.matmul ins(%m84, %W85 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e85 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m86 = dia.matmul ins(%m85, %W86 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e86 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m87 = dia.matmul ins(%m86, %W87 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e87 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m88 = dia.matmul ins(%m87, %W88 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e88 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m89 = dia.matmul ins(%m88, %W89 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e89 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m90 = dia.matmul ins(%m89, %W90 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e90 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m91 = dia.matmul ins(%m90, %W91 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e91 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m92 = dia.matmul ins(%m91, %W92 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e92 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m93 = dia.matmul ins(%m92, %W93 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e93 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m94 = dia.matmul ins(%m93, %W94 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e94 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m95 = dia.matmul ins(%m94, %W95 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e95 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m96 = dia.matmul ins(%m95, %W96 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e96 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m97 = dia.matmul ins(%m96, %W97 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e97 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m98 = dia.matmul ins(%m97, %W98 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e98 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m99 = dia.matmul ins(%m98, %W99 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e99 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %m100 = dia.matmul ins(%m99, %W100 : tensor<?x1024xf32>, tensor<1024x1024xf32>) outs(%e100 : tensor<?x1024xf32>) -> tensor<?x1024xf32>
  %index = arith.constant 0 : index
  %result = tensor.extract %m100[%index, %index] : tensor<?x1024xf32>
  return %result : f32
}
