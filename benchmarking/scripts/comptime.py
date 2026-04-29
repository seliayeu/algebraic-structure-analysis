import re
import os
import random
import time
import csv
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import logging
from custom_logging import CustomFormatter

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

for handler in logger.handlers:
    handler.setFormatter(CustomFormatter())

BUILD_DIR = Path(os.environ.get("BUILD_DIR", "./build"))
RESULT_DIR = Path("./results")
TMP_DIR = Path("/tmp/bpa_comptime/")

RESULT_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)


@dataclass
class ProgramSpec:
    num_ops: int
    tensor_size: int
    bandwidth: int
    seed: int


def run_cmd(cmd, input_text=None):
    result = subprocess.run(cmd, input=input_text, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout


def time_block(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    end = time.perf_counter()
    return out, (end - start)


def generate_chain(spec: ProgramSpec) -> str:
    random.seed(spec.seed)
    size = spec.tensor_size
    bw = spec.bandwidth

    lines = []

    lines.append("func.func @kernel() -> f32 {")
    lines.append(
        f"%input = arith.constant {{metadata = {{lowerBw = {bw} : i64, upperBw = {bw} : i64, propertyDims = [0,1]}}}} dense<1.0> : tensor<{size}x{size}xf32>"
    )

    prev = "input"

    for i in range(spec.num_ops):
        op = random.choice(["matmul", "add", "mul", "sub", "transpose"])

        buf = f"e{i}_buf"
        res = f"e{i}"

        lines.append(f"%{buf} = tensor.empty() : tensor<{size}x{size}xf32>")

        if op == "matmul":
            lines.append(
                f"%{res} = linalg.matmul ins(%{prev}, %{prev} : tensor<{size}x{size}xf32>, tensor<{size}x{size}xf32>) outs(%{buf} : tensor<{size}x{size}xf32>) -> tensor<{size}x{size}xf32>"
            )
        elif op == "add":
            lines.append(
                f"%{res} = linalg.elementwise kind=#linalg.elementwise_kind<add> ins(%{prev}, %{prev} : tensor<{size}x{size}xf32>, tensor<{size}x{size}xf32>) outs(%{buf} : tensor<{size}x{size}xf32>) -> tensor<{size}x{size}xf32>"
            )
        elif op == "mul":
            lines.append(
                f"%{res} = linalg.elementwise kind=#linalg.elementwise_kind<mul> ins(%{prev}, %{prev} : tensor<{size}x{size}xf32>, tensor<{size}x{size}xf32>) outs(%{buf} : tensor<{size}x{size}xf32>) -> tensor<{size}x{size}xf32>"
            )
        elif op == "sub":
            lines.append(
                f"%{res} = linalg.elementwise kind=#linalg.elementwise_kind<sub> ins(%{prev}, %{prev} : tensor<{size}x{size}xf32>, tensor<{size}x{size}xf32>) outs(%{buf} : tensor<{size}x{size}xf32>) -> tensor<{size}x{size}xf32>"
            )
        elif op == "transpose":
            lines.append(
                f"%{res} = linalg.transpose ins(%{prev} : tensor<{size}x{size}xf32>) outs(%{buf} : tensor<{size}x{size}xf32>) permutation = [1,0]"
            )

        prev = res

    lines.append("%c0 = arith.constant 0 : index")
    lines.append(f"%val = tensor.extract %{prev}[%c0, %c0] : tensor<{size}x{size}xf32>")
    lines.append("return %val : f32")
    lines.append("}")

    return "\n".join(lines)


LOWERING_PIPELINE = [
    "-empty-tensor-to-alloc-tensor",
    "-one-shot-bufferize=bufferize-function-boundaries",
    "-convert-linalg-to-loops",
    "-convert-scf-to-cf",
    "-convert-math-to-llvm",
    "-convert-math-to-libm",
    "-expand-strided-metadata",
    "-lower-affine",
    "-convert-arith-to-llvm",
    "-convert-func-to-llvm",
    "-convert-cf-to-llvm",
    "-finalize-memref-to-llvm",
    "-reconcile-unrealized-casts",
]


def run_mlir_passes(mlir_file: Path, passes: List[str]):
    return run_cmd([str(BUILD_DIR / "tools/alg-opt"), str(mlir_file)] + passes)


def compile_pipeline(
    mlir_file: Path, flags: List[str], warmup: int = 1, runs_count: int = 5
):
    logger.info(f"Lowering {mlir_file.name} to LLVM")

    flag_str = flags[0] if flags else ""
    flag_list = flag_str.split() if flag_str else []

    analysis_avg = 0.0
    if flag_list:
        for _ in range(warmup):
            run_mlir_passes(mlir_file, flag_list)

        analysis_times = []
        for _ in range(runs_count):
            analysis_time = rewrite_time = 0

            try:
                cmd = [str(BUILD_DIR / "tools/alg-opt"), str(mlir_file)] + flag_list
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode()
            except subprocess.CalledProcessError as e:
                print(f"Error collecting analysis time {str(e)}")

            match_analysis = re.search(r"BandedAnalysis time: ([\d.eE+-]+) ms", output)
            if match_analysis:
                analysis_time = float(match_analysis.group(1))

            match_rewrite = re.search(r"BandedRewrite time: ([\d.eE+-]+) ms", output)
            if match_rewrite:
                rewrite_time = float(match_rewrite.group(1))

            analysis_times.append(
                (analysis_time + rewrite_time) / 1000
            )  # convert to seconds

        analysis_avg = sum(analysis_times) / len(analysis_times)

    for _ in range(warmup):
        run_mlir_passes(mlir_file, flag_list + LOWERING_PIPELINE)

    total_times = []
    for _ in range(runs_count):
        _, t = time_block(run_mlir_passes, mlir_file, flag_list + LOWERING_PIPELINE)
        total_times.append(t)

    total_avg = sum(total_times) / len(total_times)

    out = run_mlir_passes(mlir_file, flag_list + LOWERING_PIPELINE)

    return out, analysis_avg, total_avg


def compile_ll(llvm_ir: str, obj_path: Path):
    logger.info(f"Compiling {obj_path.name}")
    ll_file = obj_path.with_suffix(".ll")
    ll_file.write_text(llvm_ir)
    run_cmd(
        [
            "llc",
            "-O3",
            "-relocation-model=pic",
            "-filetype=obj",
            str(ll_file),
            "-o",
            str(obj_path),
        ]
    )


def build_executable(obj_path: Path, exe_path: Path):
    logger.info(f"Building executable {exe_path.name}")

    mlir_lib_dir = os.environ.get(
        "MLIR_LIB_DIR", "/Users/kaio/projects/llvm-project/build/lib"
    )

    if not mlir_lib_dir:
        raise RuntimeError("MLIR_LIB_DIR environment variable not set")

    run_cmd(
        [
            "clang++",
            "-O3",
            "benchmarking/scripts/bench.cpp",
            str(obj_path),
            "-L",
            mlir_lib_dir,
            "-Wl,-rpath," + mlir_lib_dir,
            "-lmlir_runner_utils",
            "-lmlir_c_runner_utils",
            "-o",
            str(exe_path),
        ]
    )


def run_exe(exe_path: Path, warmup=1, runs=3):
    logger.info(f"Executing {exe_path}")
    cmd = [str(exe_path), str(warmup), str(runs)]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Benchmark failed for {exe_path.name}")
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")
        raise RuntimeError("Benchmark failed")

    out_lines = result.stdout.splitlines()

    total = float(
        next(l for l in out_lines if "total time:" in l).split(":")[1].split()[0]
    )
    avg = float(next(l for l in out_lines if "avg time:" in l).split(":")[1].split()[0])

    return total, avg


def run_experiment(
    name: str,
    configs: List[Tuple[str, List[str]]],
    sizes: List[int],
    ops_range: List[int],
    bw: int = 1,
    warmup: int = 1,
    runs_count: int = 5,
    program_dir="./benchmarking/programs",
):
    csv_path = RESULT_DIR / f"{name}.csv"
    csv_path.write_text("")
    header = False

    total_configs = len(configs) * len(ops_range)
    current = 0

    for cfg_name, flags in configs:
        logger.info("")
        logger.info(f"CONFIGURATION: {cfg_name}")
        logger.info(f"   Flags: {' '.join(flags) if flags else 'none'}")
        logger.info("-" * 60)
        for size in sizes:
            for ops in ops_range:
                current += 1

                logger.info("")
                logger.info(f"[{current}/{total_configs}] Number of ops: {ops}")

                spec = ProgramSpec(ops, size, bw, ops * 31 + size)
                mlir = generate_chain(spec)

                base = TMP_DIR / f"{cfg_name}_{size}_{ops}_{bw}"
                mlir_file = base.with_suffix(".mlir")
                mlir_file.write_text(mlir)
                mlir_out, a_t, l_t = compile_pipeline(
                    mlir_file, flags, warmup=warmup, runs_count=runs_count
                )
                llvm_ir = run_cmd(
                    ["mlir-translate", "--mlir-to-llvmir"], input_text=mlir_out
                )
                obj = base.with_suffix(".o")
                exe = base.with_suffix(".out")
                compile_ll(llvm_ir, obj)
                build_executable(obj, exe)
                runtime_total, runtime_avg = run_exe(exe, warmup, runs_count)
                row = {
                    "config": cfg_name,
                    "size": size,
                    "ops": ops,
                    "bw": bw,
                    "analysis_time_avg": a_t,
                    "total_lowering_time_avg": l_t,
                    "runtime_avg": runtime_avg,
                    "runtime_total": runtime_total,
                }
                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    if not header:
                        writer.writeheader()
                        header = True
                    writer.writerow(row)

    file_pattern = [
        ("kalman_filter_dense.mlir", "1024"),
        ("sparse_attention_dense.mlir", "1024"),
        ("batch_bertlike_dense.mlir", "2048"),
        ("chain100_dense.mlir", "1024"),
    ]

    for cfg_name, flags in configs:
        logger.info("")
        logger.info(f"CONFIGURATION: {cfg_name}")
        logger.info(f"   Flags: {' '.join(flags) if flags else 'none'}")
        logger.info("-" * 60)
        for size in sizes:
            for file_name, original_size in file_pattern:
                src = Path(program_dir) / file_name
                name = f"{file_name.replace('.mlir', '')}_{size}_{bw}"
                base = TMP_DIR / name

                mlir_file = base.with_suffix(".mlir")

                replace_file_template(src, mlir_file, "X", str(bw))
                replace_file_template(mlir_file, mlir_file, original_size, str(size))
                replace_file_template(
                    mlir_file, mlir_file, str(int(original_size) - 1), str(size - 1)
                )

                mlir_out, a_t, l_t = compile_pipeline(
                    mlir_file, flags, warmup=warmup, runs_count=runs_count
                )
                llvm_ir = run_cmd(
                    ["mlir-translate", "--mlir-to-llvmir"], input_text=mlir_out
                )
                obj = base.with_suffix(".o")
                exe = base.with_suffix(".out")
                compile_ll(llvm_ir, obj)
                build_executable(obj, exe)
                runtime_total, runtime_avg = run_exe(exe, warmup, runs_count)
                row = {
                    "config": cfg_name,
                    "size": size,
                    "ops": file_name.split("_dense")[0].replace("_", " "),
                    "bw": bw,
                    "analysis_time_avg": a_t,
                    "total_lowering_time_avg": l_t,
                    "runtime_avg": runtime_avg,
                    "runtime_total": runtime_total,
                }
                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=row.keys())
                    if not header:
                        writer.writeheader()
                        header = True
                    writer.writerow(row)

    return csv_path


def replace_file_template(src: Path, dst: Path, from_str: str, to_str: str):
    dst.write_text(src.read_text().replace(from_str, to_str))


if __name__ == "__main__":
    configs = [
        # ("baseline", []),
        # ("analysis", ["--banded-analysis"]),
        # ("analysis_detect", ["--banded-analysis=detect-dia=true"]),
        ("analysis_rewrite", ["--banded-analysis --banded-rewrite"]),
        # (
        #     "analysis_detect_rewrite",
        #     ["--banded-analysis=detect-dia=true --banded-rewrite"],
        # ),
    ]
    run_experiment("bpa_comptime", configs, [128], [20, 30, 50, 100, 200])
