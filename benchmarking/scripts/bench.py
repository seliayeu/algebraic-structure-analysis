import os
import platform
from pathlib import Path
import subprocess
import csv
from typing import List, Tuple
import logging


class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    cyan = "\x1b[36;20m"
    blue = "\x1b[34;20m"
    magenta = "\x1b[35;20m"
    bold = "\x1b[1m"
    reset = "\x1b[0m"

    def format(self, record):
        if record.levelno == logging.INFO:
            self._style._fmt = f"{self.green}[%(asctime)s]{self.reset} {self.blue}%(message)s{self.reset}"
        elif record.levelno == logging.WARNING:
            self._style._fmt = (
                f"{self.yellow}[%(asctime)s] WARNING: %(message)s{self.reset}"
            )
        elif record.levelno == logging.ERROR:
            self._style._fmt = f"{self.red}[%(asctime)s] ERROR: %(message)s{self.reset}"
        elif record.levelno == logging.DEBUG:
            self._style._fmt = (
                f"{self.grey}[%(asctime)s] DEBUG: %(message)s{self.reset}"
            )
        else:
            self._style._fmt = f"[%(asctime)s] %(message)s"

        self._style._fmt = f"{self.bold}[%(levelname)s]{self.reset} " + self._style._fmt
        return super().format(record)


logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

for handler in logger.handlers:
    handler.setFormatter(CustomFormatter())

BUILD_DIR = Path("./build")
TMP_DIR = Path("/tmp/mlir_bench")
RESULT_DIR = Path("./results")


def run_benchmark_full(exe_path: Path, warmup: int, runs: int):
    logger.debug(f"Executing benchmark: {exe_path.name}")
    system = platform.system()

    if system == "Darwin":
        cmd = ["/usr/bin/time", "-l", str(exe_path), str(warmup), str(runs)]
    else:
        cmd = ["/usr/bin/time", "-v", str(exe_path), str(warmup), str(runs)]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Benchmark failed for {exe_path.name}")
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")
        raise RuntimeError("Benchmark failed")

    out_lines = result.stdout.splitlines()
    err_lines = result.stderr.splitlines()

    total = float(
        next(l for l in out_lines if "total time:" in l).split(":")[1].split()[0]
    )
    avg = float(next(l for l in out_lines if "avg time:" in l).split(":")[1].split()[0])

    stats = {
        "total_time_s": total,
        "avg_time_s": avg,
    }

    for line in err_lines:
        line = line.strip()

        if system == "Darwin":
            if "maximum resident set size" in line:
                stats["max_rss_mb"] = int(line.split()[0]) / (1024 * 1024)
            elif "user time" in line:
                stats["user_time_s"] = float(line.split()[0])
            elif "system time" in line:
                stats["sys_time_s"] = float(line.split()[0])
            elif "page faults" in line:
                stats["page_faults"] = int(line.split()[0])
            elif "page reclaims" in line:
                stats["page_reclaims"] = int(line.split()[0])
            elif "cycles elapsed" in line:
                stats["cycles"] = int(line.split()[0])

        else:
            if "Maximum resident set size" in line:
                stats["max_rss_mb"] = int(line.split()[-1]) / 1024
            elif "User time (seconds)" in line:
                stats["user_time_s"] = float(line.split()[-1])
            elif "System time (seconds)" in line:
                stats["sys_time_s"] = float(line.split()[-1])
            elif "Percent of CPU" in line:
                stats["cpu_percent"] = line.split(":")[1].strip()
            elif "Elapsed (wall clock) time" in line:
                stats["wall_time"] = line.split(": ", 1)[1]
            elif "Major (requiring I/O) page faults" in line:
                stats["major_page_faults"] = int(line.split()[-1])
            elif "Minor (reclaiming a frame) page faults" in line:
                stats["minor_page_faults"] = int(line.split()[-1])
            elif "Voluntary context switches" in line:
                stats["vol_ctx_switches"] = int(line.split()[-1])
            elif "Involuntary context switches" in line:
                stats["invol_ctx_switches"] = int(line.split()[-1])

    logger.debug(f"Benchmark completed: {exe_path.name} - Avg time: {avg:.4f}s")
    return stats


def run_cmd(cmd, input_text=None):
    logger.debug(f"Running command: {' '.join(cmd[:3])}...")
    result = subprocess.run(cmd, input=input_text, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {' '.join(cmd)}")
        logger.error(f"STDOUT: {result.stdout}")
        logger.error(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout


def replace_bandwidth(src: Path, dst: Path, bw: int):
    dst.write_text(src.read_text().replace("X", str(bw)))


def build_pipeline_flags(cfg):
    flags = []
    tag = ""

    for c in cfg[1:]:
        if c == "analysis":
            flags.append("--banded-analysis")
            tag += "A"
        elif c == "analysis-detect":
            flags.append("--banded-analysis=detect-dia=true")
            tag += "AD"
        elif c == "rewrite":
            flags.append("--banded-rewrite")
            tag += "R"

    return flags, tag if tag else "baseline"


def lower_to_llvm(mlir_file: Path, flags: List[str]):
    logger.info(f"Lowering {mlir_file.name} to LLVM")
    cmd = (
        [f"{BUILD_DIR}/tools/alg-opt", str(mlir_file)]
        + flags
        + [
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
    )
    mlir_out = run_cmd(cmd)
    llvm_ir = run_cmd(["mlir-translate", "--mlir-to-llvmir"], input_text=mlir_out)
    return llvm_ir


def compile_kernel(ll_path: Path, obj_path: Path):
    logger.info(f"Compiling {ll_path.name}")
    run_cmd(
        [
            "llc",
            "-O3",
            "-relocation-model=pic",
            "-filetype=obj",
            str(ll_path),
            "-o",
            str(obj_path),
        ]
    )


def build_executable(obj_path: Path, exe_path: Path):
    logger.info(f"Building executable {exe_path.name}")

    mlir_lib_dir = os.environ.get("MLIR_LIB_DIR")

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


def run_benchmark(exe_path: Path, warmup: int, runs: int):
    logger.info(f"Running {exe_path.name} (warmup={warmup}, runs={runs})")
    out = run_cmd([str(exe_path), str(warmup), str(runs)])
    lines = out.splitlines()

    total = float([l for l in lines if "total time:" in l][0].split(":")[1].split()[0])
    avg = float([l for l in lines if "avg time:" in l][0].split(":")[1].split()[0])

    return total, avg


def write_csv_row(csv_path: Path, row_dict: dict, write_header: bool = False):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)
        f.flush()


def run(
    benchmark_name: str,
    program_dir: str,
    configs: List[Tuple[str]],
    bandwidths: List[int],
    warmup: int,
    runs_count: int,
) -> str:
    logger.info("=" * 80)
    logger.info(f"STARTING BENCHMARK: {benchmark_name}")
    logger.info("=" * 80)

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULT_DIR / f"{benchmark_name}.csv"
    csv_path.write_text("")
    header_written = False

    total_configs = len(configs) * len(bandwidths)
    current = 0

    for cfg in configs:
        file_name = cfg[0]
        src = Path(program_dir) / file_name
        flags, tag = build_pipeline_flags(cfg)

        logger.info("")
        logger.info(f"CONFIGURATION: {tag}")
        logger.info(f"   File: {file_name}")
        logger.info(f"   Flags: {' '.join(flags) if flags else 'none'}")
        logger.info("-" * 60)

        # No need to run for all bands
        bw_list = [0] if tag == "baseline" else bandwidths

        for bw in bw_list:
            current += 1
            logger.info("")
            logger.info(f"[{current}/{total_configs}] Testing: {tag} @ bandwidth={bw}")

            name = f"{file_name.replace('.mlir', '')}_{tag}_{bw}"

            mlir_file = TMP_DIR / f"{name}.mlir"
            ll_file = TMP_DIR / f"{name}.ll"
            obj_file = TMP_DIR / f"{name}.o"
            exe_file = TMP_DIR / f"{name}.out"

            replace_bandwidth(src, mlir_file, bw)
            logger.debug(f"   Generated MLIR: {mlir_file.name}")

            llvm_ir = lower_to_llvm(mlir_file, flags)
            ll_file.write_text(llvm_ir)

            compile_kernel(ll_file, obj_file)
            build_executable(obj_file, exe_file)

            stats = run_benchmark_full(exe_file, warmup, runs_count)

            row = {"file_name": file_name, "config": tag, "bw": bw, **stats}
            write_csv_row(csv_path, row, write_header=not header_written)
            header_written = True

            logger.info(
                f"[{current}/{total_configs}] Completed: {tag} @ bw={bw} (avg={stats['avg_time_s']:.4f}s)"
            )

    logger.info("")
    logger.info("=" * 80)
    logger.info(f"BENCHMARK COMPLETED: {benchmark_name}")
    logger.info(f"Results saved to: {csv_path}")
    logger.info("=" * 80)

    return csv_path


if __name__ == "__main__":
    bandwidths = [512, 409, 307, 256, 204, 153, 102, 51, 0]

    configs = [
        ("chain_matmul_dense.mlir", None),
        ("chain_matmul_dense.mlir", "analysis", "rewrite"),
        ("chain_matmul_dia.mlir", "analysis", "rewrite"),
        ("chain_matmul_dia.mlir", "analysis-detect", "rewrite"),
    ]

    run(
        benchmark_name="chain_matmul",
        program_dir="./benchmarking/programs",
        configs=configs,
        bandwidths=bandwidths,
        warmup=1,
        runs_count=1,
    )

    bandwidths = [1024, 512, 409, 307, 256, 204, 153, 102, 51, 0]
    configs_bert = [
        ("bertlike_dia.mlir", "analysis", "rewrite"),
        ("bertlike_dia.mlir", "analysis-detect", "rewrite"),
        ("bertlike_dense.mlir", None),
        ("bertlike_dense.mlir", "analysis", "rewrite"),
    ]

    run(
        benchmark_name="bertlike",
        program_dir="./benchmarking/programs",
        configs=configs_bert,
        bandwidths=bandwidths,
        warmup=1,
        runs_count=5,
    )
