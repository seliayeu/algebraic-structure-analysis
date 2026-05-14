#!/usr/bin/env python3
"""Generate end-to-end correctness lit tests for K-chained matmul programs.

Each variant writes an explicit 2-D array literal so the IR is semantically
correct: zeros are present outside the bandwidth, and DIA-format weight tensors
carry the proper leading/trailing padding per diagonal.

Conventions (matches your example):
  - DIA storage: rows ordered d = -lowerBw, ..., 0, ..., +upperBw
  - storage[d_idx][i] = dense[i][i+d],  d = d_idx - lowerBw
  - Leading zeros for i+d < 0 (lower diagonals at small i)
  - Trailing zeros for i+d >= N (upper diagonals at large i)

Usage:
    python gen_chain_tests.py --K 3 --N 5 --bandwidth 2 --integer-weights \\
        --out-dir ./tests/generated
"""

import argparse, os
import numpy as np


# ---------------------------------------------------------------------------
# RUN block
# ---------------------------------------------------------------------------

def _run_block(detect_dia: bool) -> str:
    head = '--banded-analysis="detect-dia=true"' if detect_dia else '--banded-analysis'
    pipeline = (
        f'{head} --banded-rewrite '
        '--reconcile-unrealized-casts '
        '--one-shot-bufferize="allow-return-allocs-from-loops=true '
        'bufferize-function-boundaries=true" '
        '--convert-linalg-to-loops --convert-scf-to-cf --convert-cf-to-llvm '
        '--convert-arith-to-llvm --finalize-memref-to-llvm --convert-func-to-llvm '
        '--reconcile-unrealized-casts'
    )
    return (
        f'// RUN: %build/tools/alg-opt %s {pipeline} | \\\n'
        '// RUN:   mlir-runner --entry-point-result=void \\\n'
        '// RUN:     --shared-libs=%llvm_root/build/lib/'
        'libmlir_runner_utils.%lib_format > %t\n'
        '// RUN: FileCheck %s < %t\n'
    )


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def _banded_matrix(N: int, bw: int, fill: float) -> np.ndarray:
    """N×N matrix, fill in band, 0 outside."""
    M = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(max(0, i - bw), min(N, i + bw + 1)):
            M[i, j] = fill
    return M


def _to_dia(M: np.ndarray, lowerBw: int, upperBw: int) -> np.ndarray:
    """Dense N×N -> DIA storage (lowerBw+upperBw+1) × N.

    Row 0 = d=-lowerBw (most-negative diagonal, leading zeros at small i).
    Row lowerBw = d=0 (main diagonal).
    Row lowerBw+upperBw = d=+upperBw (trailing zeros at large i).
    """
    N = M.shape[0]
    nrows = lowerBw + upperBw + 1
    out = np.zeros((nrows, N), dtype=M.dtype)
    for d_idx in range(nrows):
        d = d_idx - lowerBw
        for i in range(N):
            j = i + d
            if 0 <= j < N:
                out[d_idx, i] = M[i, j]
    return out


def _chain_result(K: int, N: int, bw: int, weights: list) -> np.ndarray:
    R = _banded_matrix(N, bw, 1.0)
    for w in weights:
        R = R @ _banded_matrix(N, bw, w)
    return R.astype(np.float32)


# ---------------------------------------------------------------------------
# MLIR literal formatting
# ---------------------------------------------------------------------------

def _fmt_val(v: float) -> str:
    """Always emit a decimal point so MLIR accepts the constant for f32."""
    iv = int(v)
    return f'{iv}.0' if v == iv else f'{v:.4f}'


def _array2d_lit(M: np.ndarray) -> str:
    rows = ', '.join(
        '[' + ', '.join(_fmt_val(v) for v in row) + ']'
        for row in M
    )
    return f'dense<[{rows}]>'


def _md(lowerBw: int, upperBw: int, dia: bool = False) -> str:
    dia_str = 'dia = true, ' if dia else ''
    return (
        f'{{metadata = {{{dia_str}'
        f'lowerBw = {lowerBw} : i64, upperBw = {upperBw} : i64, '
        f'propertyDims = [0, 1]}}}}'
    )


# ---------------------------------------------------------------------------
# CHECK formatting
# ---------------------------------------------------------------------------

def _cell(v: float) -> str:
    """FileCheck pattern robust to %g formatting.

    mlir-runner uses C++ default << for float: plain decimal when |v| < 1e6,
    scientific (e.g. 1.10125e+06) otherwise. Take a 4-char prefix of the %.6g
    representation so the pattern survives both formats and fp rounding drift.
    """
    if v == 0:
        return '0{{.*}}'
    s = f'{float(v):.6g}'
    return f'{s[:4]}{{{{.*}}}}'


def _checks(M: np.ndarray) -> list:
    rows, cols = M.shape
    out = [
        f'// CHECK: Unranked Memref {{{{.*}}}} rank = 2 offset = 0 '
        f'sizes = [{rows}, {cols}]'
    ]
    for r, row in enumerate(M):
        body = ', '.join(_cell(v) for v in row)
        opener = '{{\\[}}[' if r == 0 else '['
        closer = ']]' if r == rows - 1 else '],'
        directive = '// CHECK:' if r == 0 else '// CHECK-NEXT:'
        out.append(f'{directive} {opener}{body}{closer}')
    return out


# ---------------------------------------------------------------------------
# IR chain builders
# ---------------------------------------------------------------------------

def _dense_chain(K: int, N: int, bw: int, weights: list):
    """linalg.matmul chain; weights are N×N with explicit zeros outside band."""
    t = f'tensor<{N}x{N}xf32>'
    md = _md(bw, bw)
    L = []
    for i in range(1, K + 1):
        L.append(f'    %e{i} = arith.constant dense<0.0> : {t}')
    L.append(f'    %input = arith.constant {md}')
    L.append(f'        {_array2d_lit(_banded_matrix(N, bw, 1.0))} : {t}')
    for i, w in enumerate(weights, start=1):
        L.append(f'    %W{i} = arith.constant {md}')
        L.append(f'        {_array2d_lit(_banded_matrix(N, bw, w))} : {t}')
    L.append(
        f'    %m1 = linalg.matmul ins(%input, %W1 : {t}, {t}) '
        f'outs(%e1 : {t}) -> {t}'
    )
    for i in range(2, K + 1):
        L.append(
            f'    %m{i} = linalg.matmul ins(%m{i-1}, %W{i} : {t}, {t}) '
            f'outs(%e{i} : {t}) -> {t}'
        )
    return '\n'.join(L), f'%m{K}', t


def _dia_chain(K: int, N: int, bw: int, weights: list):
    """dia.matmul chain; weights are N×N dense-with-metadata, result is DIA (?×N)."""
    wt = f'tensor<{N}x{N}xf32>'
    ot = f'tensor<?x{N}xf32>'
    md = _md(bw, bw)
    L = ['    %c0 = arith.constant 0 : index']
    for i in range(1, K + 1):
        L.append(f'    %e{i} = tensor.empty(%c0) : {ot}')
    L.append(f'    %input = arith.constant {md}')
    L.append(f'        {_array2d_lit(_banded_matrix(N, bw, 1.0))} : {wt}')
    for i, w in enumerate(weights, start=1):
        L.append(f'    %W{i} = arith.constant {md}')
        L.append(f'        {_array2d_lit(_banded_matrix(N, bw, w))} : {wt}')
    L.append(
        f'    %m1 = dia.matmul ins(%input, %W1 : {wt}, {wt}) '
        f'outs(%e1 : {ot}) -> {ot}'
    )
    for i in range(2, K + 1):
        L.append(
            f'    %m{i} = dia.matmul ins(%m{i-1}, %W{i} : {ot}, {wt}) '
            f'outs(%e{i} : {ot}) -> {ot}'
        )
    return '\n'.join(L), f'%m{K}', ot


def _dia_inputs_chain(K: int, N: int, bw: int, weights: list):
    """dia.matmul chain; weights are in explicit DIA storage ((2*bw+1)×N)."""
    storage_dim = 2 * bw + 1
    wt = f'tensor<{storage_dim}x{N}xf32>'
    ot = f'tensor<?x{N}xf32>'
    md = _md(bw, bw, dia=True)
    L = ['    %c0 = arith.constant 0 : index']
    for i in range(1, K + 1):
        L.append(f'    %e{i} = tensor.empty(%c0) : {ot}')
    L.append(f'    %input = arith.constant {md}')
    L.append(f'        {_array2d_lit(_to_dia(_banded_matrix(N, bw, 1.0), bw, bw))} : {wt}')
    for i, w in enumerate(weights, start=1):
        W_dia = _to_dia(_banded_matrix(N, bw, w), bw, bw)
        L.append(f'    %W{i} = arith.constant {md}')
        L.append(f'        {_array2d_lit(W_dia)} : {wt}')
    L.append(
        f'    %m1 = dia.matmul ins(%input, %W1 : {wt}, {wt}) '
        f'outs(%e1 : {ot}) -> {ot}'
    )
    for i in range(2, K + 1):
        L.append(
            f'    %m{i} = dia.matmul ins(%m{i-1}, %W{i} : {ot}, {wt}) '
            f'outs(%e{i} : {ot}) -> {ot}'
        )
    return '\n'.join(L), f'%m{K}', ot


# ---------------------------------------------------------------------------
# @main wrapper
# ---------------------------------------------------------------------------

def _wrap_main(body: str, ssa: str, ttype: str, dynamic: bool) -> str:
    mtype = ttype.replace('tensor', 'memref')
    if dynamic:
        return (
            '  func.func @main() {\n'
            f'{body}\n'
            f'    %dim = tensor.dim {ssa}, %c0 : {ttype}\n'
            f'    %memref = memref.alloc(%dim) : {mtype}\n'
            f'    bufferization.materialize_in_destination {ssa} in writable %memref\n'
            f'        : ({ttype}, {mtype}) -> ()\n'
            f'    %cast = memref.cast %memref : {mtype} to memref<*xf32>\n'
            '    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()\n'
            f'    memref.dealloc %memref : {mtype}\n'
            '    return\n'
            '  }'
        )
    return (
        '  func.func @main() {\n'
        f'{body}\n'
        f'    %memref = memref.alloc() : {mtype}\n'
        f'    bufferization.materialize_in_destination {ssa} in %memref {{writable}}\n'
        f'        : ({ttype}, {mtype}) -> ()\n'
        f'    %cast = memref.cast %memref : {mtype} to memref<*xf32>\n'
        '    call @printMemrefF32(%cast) : (memref<*xf32>) -> ()\n'
        f'    memref.dealloc %memref : {mtype}\n'
        '    return\n'
        '  }'
    )


# ---------------------------------------------------------------------------
# Per-variant assembly
# ---------------------------------------------------------------------------

def _emit(variant: str, K: int, N: int, bw: int, weights: list) -> str:
    dense_result = _chain_result(K, N, bw, weights)
    total_bw = min((K + 1) * bw, N - 1)

    if variant == 'dense':
        body, ssa, t = _dense_chain(K, N, bw, weights)
        expected = dense_result        # N×N
        dynamic = False
        detect_dia = True
    elif variant == 'dia':
        body, ssa, t = _dia_chain(K, N, bw, weights)
        expected = _to_dia(dense_result, total_bw, total_bw)  # (2*tb+1)×N
        dynamic = True
        detect_dia = False
    else:  # dia_inputs
        body, ssa, t = _dia_inputs_chain(K, N, bw, weights)
        expected = _to_dia(dense_result, total_bw, total_bw)
        dynamic = True
        detect_dia = False

    return (
        _run_block(detect_dia)
        + '//\n'
        + f'// Auto-generated correctness test: '
          f'K={K}, N={N}, bw={bw}, variant={variant}.\n'
        + '//\n'
        + '\n'.join(_checks(expected)) + '\n'
        + 'module {\n'
        + '  func.func private @printMemrefF32(memref<*xf32>)\n'
        + _wrap_main(body, ssa, t, dynamic) + '\n'
        + '}\n'
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='tests/generated')
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--N', type=int, default=10)
    ap.add_argument('--bandwidth', type=int, default=2)
    ap.add_argument('--variants', nargs='+',
                    choices=['dense', 'dia', 'dia_inputs'],
                    default=['dense', 'dia', 'dia_inputs'])
    ap.add_argument('--integer-weights', action='store_true',
                    help='W_i = float(i). Produces integer-valued results '
                         'so values never hit scientific notation for small K/N.')
    args = ap.parse_args()

    weights = (
        [float(i) for i in range(1, args.K + 1)]
        if args.integer_weights
        else [1.0 + 0.01 * i for i in range(1, args.K + 1)]
    )

    os.makedirs(args.out_dir, exist_ok=True)
    for v in args.variants:
        path = os.path.join(args.out_dir, f'chain{args.K}_{v}_correctness.mlir')
        with open(path, 'w') as f:
            f.write(_emit(v, args.K, args.N, args.bandwidth, weights))
        print(f'wrote {path}')


if __name__ == '__main__':
    main()
