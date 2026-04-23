import platform
from pathlib import Path
import subprocess
import csv
import logging
from itertools import product
from typing import List

# ── Logging ───────────────────────────────────────────────────────────────────

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"; yellow = "\x1b[33;20m"; red = "\x1b[31;20m"
    green = "\x1b[32;20m"; blue = "\x1b[34;20m"; bold = "\x1b[1m"; reset = "\x1b[0m"

    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = f"{self.green}[%(asctime)s]{self.reset} {self.blue}%(message)s{self.reset}"
        elif record.levelno == logging.WARNING:
            self._style._fmt = f"{self.yellow}[%(asctime)s] WARNING: %(message)s{self.reset}"
        elif record.levelno == logging.ERROR:
            self._style._fmt = f"{self.red}[%(asctime)s] ERROR: %(message)s{self.reset}"
        else:
            self._style._fmt = f"{self.grey}[%(asctime)s] DEBUG: %(message)s{self.reset}"
        self._style._fmt = f"{self.bold}[%(levelname)s]{self.reset} " + self._style._fmt
        return super().format(record)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
for handler in logger.handlers:
    handler.setFormatter(CustomFormatter())

# ── Paths ─────────────────────────────────────────────────────────────────────

BUILD_DIR  = Path("./build")
TMP_DIR    = Path("/tmp/mlir_bench")
RESULT_DIR = Path("./results")

MATRIX_SIZE = 2048
INT64_MAX = 9223372036854775807  # used only for tensor.empty placeholder

def output_bw(bw: int) -> int:
    """Output bandwidth is always lhs_bw + rhs_bw regardless of storage format."""
    return bw + bw

def input_meta(dia: bool, bw: int) -> str:
    dia_attr = "dia = true, " if dia else ""
    return f"{{metadata = {{{dia_attr}lowerBw = {bw} : i64, propertyDims = [0, 1], upperBw = {bw} : i64}}}}"

def empty_meta() -> str:
    return f"{{metadata = {{lowerBw = {INT64_MAX} : i64, propertyDims = [0, 1], upperBw = {INT64_MAX} : i64}}}}"

def result_meta(dia: bool, bw: int) -> str:
    dia_attr = "dia = true, " if dia else ""
    return f"{{metadata = {{{dia_attr}lowerBw = {bw} : i64, propertyDims = [0, 1], upperBw = {bw} : i64}}}}"

def tensor_type(dia: bool, bw: int) -> str:
    N = MATRIX_SIZE
    if dia:
        return f"tensor<{2 * bw + 1}x{N}xf32>"
    return f"tensor<{N}x{N}xf32>"

FROM_DENSE_COMBOS = [
    # (name,              lhs_from_dense, rhs_from_dense)
    # ("dense_fd_x_dia_x_dia",        True,  False),  # dense*dia->dia via from_dense on lhs
    # ("dense_fd_x_dense_fd_x_dia",   True,  True),   # dense*dense->dia via from_dense on both
    ("dia_x_dense_fd_x_dia",   False,  True),   # dense*dense->dia via from_dense on both
]


def make_from_dense_matmul_mlir(lhs_from_dense: bool, rhs_from_dense: bool, bw: int) -> str:
    """
    Both inputs are stored dense (2048x2048) with bandwidth metadata.
    Those marked from_dense are converted via dia.from_dense before the matmul,
    making the effective computation dia * dia -> dia.
    """
    out_bw  = bw + bw
    dense_t = tensor_type(dia=False, bw=bw)   # tensor<2048x2048xf32>
    dia_t   = tensor_type(dia=True,  bw=bw)   # tensor<Yx2048xf32>
    out_t   = tensor_type(dia=True,  bw=out_bw)

    def input_block(var: str, from_dense: bool) -> str:
        meta = input_meta(dia=False, bw=bw)   # no dia=true on the constant
        lines = [f"  %{var}_dense = arith.constant {meta} dense<1.0> : {dense_t}"]
        if from_dense:
            lines.append(f"  %{var} = dia.from_dense %{var}_dense "
                f"{input_meta(dia=True, bw=bw)} : {dense_t} -> {dia_t}"
            )
        else:
            lines.append(f"  %{var} = arith.constant {input_meta(dia=True, bw=bw)} dense<1.0> : {dia_t}")
        return "\n".join(lines)

    lhs_t = dia_t if lhs_from_dense else dia_t   # after conversion both are dia_t
    rhs_t = dia_t

    return f"""\
func.func @kernel() -> f32 {{
  %out    = tensor.empty() {empty_meta()} : {out_t}
{input_block("lhs", lhs_from_dense)}
{input_block("rhs", rhs_from_dense)}
  %result = dia.matmul ins(%lhs, %rhs : {lhs_t}, {rhs_t})
                       outs(%out : {out_t}) -> {out_t} {result_meta(dia=True, bw=out_bw)}
  %index  = arith.constant 0 : index
  %val    = tensor.extract %result[%index, %index] : {out_t}
  return %val : f32
}}
"""

def make_single_matmul_mlir(lhs_dia: bool, rhs_dia: bool, out_dia: bool, bw: int) -> str:
    out_bw   = output_bw(bw)
    lhs_t    = tensor_type(lhs_dia, bw)
    rhs_t    = tensor_type(rhs_dia, bw)
    out_t    = tensor_type(out_dia, out_bw)
    res_meta = result_meta(out_dia, out_bw)

    return f"""\
func.func @kernel() -> f32 {{
  %out    = tensor.empty() {empty_meta()} : {out_t}
  %lhs    = arith.constant {input_meta(lhs_dia, bw)} dense<1.0> : {lhs_t}
  %rhs    = arith.constant {input_meta(rhs_dia, bw)} dense<2.0> : {rhs_t}
  %result = dia.matmul ins(%lhs, %rhs : {lhs_t}, {rhs_t})
                       outs(%out : {out_t}) -> {out_t} {res_meta}
  %index  = arith.constant 0 : index
  %val    = tensor.extract %result[%index, %index] : {out_t}
  return %val : f32
}}
"""
def combo_name(lhs_dia: bool, rhs_dia: bool, out_dia: bool) -> str:
    return "_x_".join("dia" if d else "dense" for d in (lhs_dia, rhs_dia, out_dia))

# ── Infrastructure ────────────────────────────────────────────────────────────

def run_cmd(cmd, input_text=None):
    result = subprocess.run(cmd, input=input_text, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {' '.join(cmd)}")
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout

def lower_to_llvm(mlir_file: Path):
    # No --banded-analysis; rewrite pass uses the embedded metadata directly.
    cmd = [
        f"{BUILD_DIR}/tools/alg-opt", str(mlir_file),
        "--canonicalize",
        "--banded-rewrite",
        "-empty-tensor-to-alloc-tensor",
        "-one-shot-bufferize=bufferize-function-boundaries",
        "-convert-linalg-to-loops",
        "-convert-scf-to-cf",
        "-expand-strided-metadata",
        "-lower-affine",
        "-convert-arith-to-llvm",
        "-convert-func-to-llvm",
        "-convert-cf-to-llvm",
        "-finalize-memref-to-llvm",
        "-reconcile-unrealized-casts",
    ]
    mlir_out = run_cmd(cmd)
    return run_cmd(["mlir-translate", "--mlir-to-llvmir"], input_text=mlir_out)

def compile_kernel(ll_path: Path, obj_path: Path):
    run_cmd(["llc", "-O3", "-relocation-model=pic", "-filetype=obj", str(ll_path), "-o", str(obj_path)])

def build_executable(obj_path: Path, exe_path: Path):
    run_cmd(["clang++", "-O3", "benchmarking/scripts/bench.cpp", str(obj_path), "-o", str(exe_path)])

def run_benchmark_full(exe_path: Path, warmup: int, runs: int) -> dict:
    system = platform.system()
    cmd = ["/usr/bin/time", "-l" if system == "Darwin" else "-v",
           str(exe_path), str(warmup), str(runs)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Benchmark failed for {exe_path.name}\n{result.stderr}")

    out_lines = result.stdout.splitlines()
    err_lines = result.stderr.splitlines()

    total = float(next(l for l in out_lines if "total time:" in l).split(":")[1].split()[0])
    avg   = float(next(l for l in out_lines if "avg time:"   in l).split(":")[1].split()[0])
    stats = {"total_time_s": total, "avg_time_s": avg}

    for line in err_lines:
        line = line.strip()
        if system == "Darwin":
            if "maximum resident set size" in line: stats["max_rss_mb"]  = int(line.split()[0]) / (1024 * 1024)
            elif "user time"   in line:             stats["user_time_s"] = float(line.split()[0])
            elif "system time" in line:             stats["sys_time_s"]  = float(line.split()[0])
        else:
            if   "Maximum resident set size"      in line: stats["max_rss_mb"]        = int(line.split()[-1]) / 1024
            elif "User time (seconds)"            in line: stats["user_time_s"]        = float(line.split()[-1])
            elif "System time (seconds)"          in line: stats["sys_time_s"]         = float(line.split()[-1])
            elif "Voluntary context switches"     in line: stats["vol_ctx_switches"]   = int(line.split()[-1])
            elif "Involuntary context switches"   in line: stats["invol_ctx_switches"] = int(line.split()[-1])
    return stats

def write_csv_row(csv_path: Path, row: dict, write_header: bool = False):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()

# ── Main benchmark loop ───────────────────────────────────────────────────────

def run_matmul_combinations(bandwidths: List[int], warmup: int, runs_count: int) -> Path:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULT_DIR / "matmul_combinations.csv"
    csv_path.write_text("")
    header_written = False

    combos  = list(product([True, False], repeat=3))
    total   = len(combos) * len(bandwidths)
    current = 0

    logger.info("=" * 80)
    logger.info(f"MATMUL COMBINATION BENCHMARK  ({len(combos)} combos × {len(bandwidths)} bws = {total} runs)")
    logger.info("=" * 80)

    # for lhs_dia, rhs_dia, out_dia in combos:
    #     if (not lhs_dia and rhs_dia and out_dia):
    #         continue
    #     name   = combo_name(lhs_dia, rhs_dia, out_dia)
    #     logger.info(f"\nCOMBO: {name}")
    #     logger.info("-" * 60)
    #
    #     for bw in bandwidths:
    #         current += 1
    #         run_name = f"matmul_{name}_bw{bw}"
    #         logger.info(f"[{current}/{total}] {run_name}")
    #
    #         out_bw_val = output_bw(bw)
    #         logger.info(f"  lhs_bw={bw}  rhs_bw={bw}  out_bw={'dense' if out_bw_val == INT64_MAX else out_bw_val}")
    #
    #         mlir_file = TMP_DIR / f"{run_name}.mlir"
    #         ll_file   = TMP_DIR / f"{run_name}.ll"
    #         obj_file  = TMP_DIR / f"{run_name}.o"
    #         exe_file  = TMP_DIR / f"{run_name}.out"
    #
    #         mlir_file.write_text(make_single_matmul_mlir(lhs_dia, rhs_dia, out_dia, bw))
    #
    #         llvm_ir = lower_to_llvm(mlir_file)
    #         ll_file.write_text(llvm_ir)
    #         compile_kernel(ll_file, obj_file)
    #         build_executable(obj_file, exe_file)
    #
    #         stats = run_benchmark_full(exe_file, warmup, runs_count)
    #
    #         row = {
    #             "lhs":    "dia" if lhs_dia else "dense",
    #             "rhs":    "dia" if rhs_dia else "dense",
    #             "out":    "dia" if out_dia else "dense",
    #             "bw":     bw,
    #             "out_bw": out_bw_val,
    #             **stats,
    #         }
    #         write_csv_row(csv_path, row, write_header=not header_written)
    #         header_written = True
    #
    #         logger.info(f"  → avg={stats['avg_time_s']:.4f}s  rss={stats.get('max_rss_mb', '?'):.1f}MB")
    #
    # # ── from_dense cases ──────────────────────────────────────────────────────
    total_fd = len(FROM_DENSE_COMBOS) * len(bandwidths)
    current_fd = 0

    logger.info("\n" + "=" * 80)
    logger.info(f"FROM_DENSE CASES  ({len(FROM_DENSE_COMBOS)} combos × {len(bandwidths)} bws = {total_fd} runs)")
    logger.info("=" * 80)

    for name, lhs_fd, rhs_fd in FROM_DENSE_COMBOS:
        logger.info(f"\nCOMBO: {name}")
        logger.info("-" * 60)

        for bw in bandwidths:
            current_fd += 1
            out_bw_val = bw + bw
            run_name = f"matmul_{name}_bw{bw}"
            logger.info(f"[{current_fd}/{total_fd}] {run_name}  out_bw={out_bw_val}")

            mlir_file = TMP_DIR / f"{run_name}.mlir"
            ll_file   = TMP_DIR / f"{run_name}.ll"
            obj_file  = TMP_DIR / f"{run_name}.o"
            exe_file  = TMP_DIR / f"{run_name}.out"

            mlir_file.write_text(make_from_dense_matmul_mlir(lhs_fd, rhs_fd, bw))

            llvm_ir = lower_to_llvm(mlir_file)
            ll_file.write_text(llvm_ir)
            compile_kernel(ll_file, obj_file)
            build_executable(obj_file, exe_file)

            stats = run_benchmark_full(exe_file, warmup, runs_count)

            row = {
                "lhs":    "dense_fd" if lhs_fd else "dia",
                "rhs":    "dense_fd" if rhs_fd else "dia",
                "out":    "dia",
                "bw":     bw,
                "out_bw": out_bw_val,
                **stats,
            }
            write_csv_row(csv_path, row, write_header=not header_written)
            header_written = True

            logger.info(f"  → avg={stats['avg_time_s']:.4f}s  rss={stats.get('max_rss_mb', '?'):.1f}MB")

    logger.info(f"\n{'=' * 80}")
    logger.info(f"DONE — results: {csv_path}")
    logger.info("=" * 80)
    return csv_path


if __name__ == "__main__":
    run_matmul_combinations(
        bandwidths=[512, 409, 307, 256, 204, 153, 102, 51],
        warmup=10,
        runs_count=1,
    )
