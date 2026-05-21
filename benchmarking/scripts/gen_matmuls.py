#!/usr/bin/env python3
"""Generate K-chained matmul programs in three variants: dia, dia-inputs, dense.

Linear chain: m_1 = matmul(input, W_1); m_i = matmul(m_{i-1}, W_i) for i in [2, K].
Non-batched, square N x N matrices.

Placeholders left intact for downstream substitution:
  X = bandwidth (lowerBw / upperBw values in metadata)
  Y = DIA storage dim (only appears in dia-inputs variant)

Empties and intermediate result tensors use `?` for the DIA storage dim.

Usage:
    python gen_chain.py --out-dir ./out [--K 100] [--N 2048]
"""
import argparse
import os


def dia_program(K: int, N: int) -> str:
    md = "{lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}"
    weight_t = f"tensor<{N}x{N}xf32>"
    out_t = f"tensor<?x{N}xf32>"

    lines = ["func.func @kernel() -> f32 {"]
    lines.append("  %c0 = arith.constant 0 : index")
    lines.append("  %cf1 = arith.constant 1.0 : f32")
    for i in range(1, K + 1):
        lines.append(f"  %e{i} = tensor.empty(%c0) : {out_t}")
    lines.append(f"  %input_empty = tensor.empty() : {weight_t}")
    for i in range(1, K + 1):
        lines.append(f"  %W{i}_empty = tensor.empty() : {weight_t}")
    lines.append(f"  %input = linalg.fill {{metadata = {md}}} ins(%cf1 : f32) outs(%input_empty : {weight_t}) -> {weight_t}")
    for i in range(1, K + 1):
        lines.append(f"  %W{i} = linalg.fill {{metadata = {md}}} ins(%cf1 : f32) outs(%W{i}_empty : {weight_t}) -> {weight_t}")
    lines.append(
        f"  %m1 = dia.matmul ins(%input, %W1 : {weight_t}, {weight_t}) "
        f"outs(%e1 : {out_t}) -> {out_t}"
    )
    for i in range(2, K + 1):
        lines.append(
            f"  %m{i} = dia.matmul ins(%m{i-1}, %W{i} : {out_t}, {weight_t}) "
            f"outs(%e{i} : {out_t}) -> {out_t}"
        )
    lines.append("  %index = arith.constant 0 : index")
    lines.append(f"  %result = tensor.extract %m{K}[%index, %index] : {out_t}")
    lines.append("  return %result : f32")
    lines.append("}")
    return "\n".join(lines) + "\n"


def dia_inputs_program(K: int, N: int) -> str:
    md = "{dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}"
    weight_t = f"tensor<Yx{N}xf32>"
    out_t = f"tensor<?x{N}xf32>"

    lines = ["func.func @kernel() -> f32 {"]
    lines.append("  %c0 = arith.constant 0 : index")
    lines.append("  %cf1 = arith.constant 1.0 : f32")
    for i in range(1, K + 1):
        lines.append(f"  %e{i} = tensor.empty(%c0) : {out_t}")
    lines.append(f"  %input_empty = tensor.empty() : {weight_t}")
    for i in range(1, K + 1):
        lines.append(f"  %W{i}_empty = tensor.empty() : {weight_t}")
    lines.append(f"  %input = linalg.fill {{metadata = {md}}} ins(%cf1 : f32) outs(%input_empty : {weight_t}) -> {weight_t}")
    for i in range(1, K + 1):
        lines.append(f"  %W{i} = linalg.fill {{metadata = {md}}} ins(%cf1 : f32) outs(%W{i}_empty : {weight_t}) -> {weight_t}")
    lines.append(
        f"  %m1 = dia.matmul ins(%input, %W1 : {weight_t}, {weight_t}) "
        f"outs(%e1 : {out_t}) -> {out_t}"
    )
    for i in range(2, K + 1):
        lines.append(
            f"  %m{i} = dia.matmul ins(%m{i-1}, %W{i} : {out_t}, {weight_t}) "
            f"outs(%e{i} : {out_t}) -> {out_t}"
        )
    lines.append("  %index = arith.constant 0 : index")
    lines.append(f"  %result = tensor.extract %m{K}[%index, %index] : {out_t}")
    lines.append("  return %result : f32")
    lines.append("}")
    return "\n".join(lines) + "\n"


def dense_program(K: int, N: int) -> str:
    md = "{lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}"
    t = f"tensor<{N}x{N}xf32>"

    lines = ["func.func @kernel() -> f32 {"]
    lines.append("  %cf1 = arith.constant 1.0 : f32")
    for i in range(1, K + 1):
        lines.append(f"  %e{i} = arith.constant dense<0.0> : {t}")
    lines.append(f"  %input_empty = tensor.empty() : {t}")
    for i in range(1, K + 1):
        lines.append(f"  %W{i}_empty = tensor.empty() : {t}")
    lines.append(f"  %input = linalg.fill {{metadata = {md}}} ins(%cf1 : f32) outs(%input_empty : {t}) -> {t}")
    for i in range(1, K + 1):
        lines.append(f"  %W{i} = linalg.fill {{metadata = {md}}} ins(%cf1 : f32) outs(%W{i}_empty : {t}) -> {t}")
    lines.append(
        f"  %m1 = linalg.matmul ins(%input, %W1 : {t}, {t}) "
        f"outs(%e1 : {t}) -> {t}"
    )
    for i in range(2, K + 1):
        lines.append(
            f"  %m{i} = linalg.matmul ins(%m{i-1}, %W{i} : {t}, {t}) "
            f"outs(%e{i} : {t}) -> {t}"
        )
    lines.append("  %index = arith.constant 0 : index")
    lines.append(f"  %result = tensor.extract %m{K}[%index, %index] : {t}")
    lines.append("  return %result : f32")
    lines.append("}")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="benchmarking/programs")
    p.add_argument("--K", type=int, default=10, help="chain length")
    p.add_argument("--N", type=int, default=10, help="matrix size")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    variants = {
        f"chain{args.K}_dia.mlir": dia_program(args.K, args.N),
        f"chain{args.K}_dia_inputs.mlir": dia_inputs_program(args.K, args.N),
        f"chain{args.K}_dense.mlir": dense_program(args.K, args.N),
    }
    for name, body in variants.items():
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as f:
            f.write(body)
        print(f"wrote {path} ({len(body.splitlines())} lines)")


if __name__ == "__main__":
    main()
