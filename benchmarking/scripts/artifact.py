import argparse
import inspect
import os
import shutil
import subprocess
from pathlib import Path

import figures
import comptime
import bench
import attention_prop


def kalman_filter(runs_count: int = 5) -> str:
    benchmark_name = "kalman_filter"
    bandwidths = [1023, 512, 256, 128, 64, 32, 16, 0]

    # D = --banded-analysis=detect-dia=true
    # A = --banded-analysis
    # R = --banded-rewrite
    # L = --linalg-ikj-loop
    # S = --dense-softmax-rewrite
    configs = [
        ("kalman_filter_dense.mlir", None),
        ("kalman_filter_dense.mlir", "L"),
        ("kalman_filter_dense.mlir", "ARL"),
        ("kalman_filter_dia.mlir", "ARL"),
        ("kalman_filter_dia.mlir", "DRL"),
        ("kalman_filter_dia_inputs.mlir", "ARL"),
        ("kalman_filter_dia_inputs.mlir", "DRL"),
    ]

    result_path = bench.run(
        benchmark_name=benchmark_name,
        program_dir="./benchmarking/programs",
        configs=configs,
        bandwidths=bandwidths,
        warmup=1,
        runs_count=runs_count,
    )

    return result_path


def batch_bertlike(runs_count: int = 5, plot=True) -> str:
    benchmark_name = "batch_bertlike"
    bandwidths = [1023, 512, 256, 128, 64, 32, 16, 0]

    configs = [
        ("batch_bertlike_dense.mlir", None),
        ("batch_bertlike_dense.mlir", "L"),
        ("batch_bertlike_dense.mlir", "ARL"),
        ("batch_bertlike_dia.mlir", "ARL"),
        ("batch_bertlike_dia.mlir", "DRL"),
        ("batch_bertlike_dia_inputs.mlir", "ARL"),
        ("batch_bertlike_dia_inputs.mlir", "DRL"),
    ]

    result_path = bench.run(
        benchmark_name=benchmark_name,
        program_dir="./benchmarking/programs",
        configs=configs,
        bandwidths=bandwidths,
        warmup=1,
        runs_count=runs_count,
    )

    return result_path


def sparse_attention(runs_count: int = 5) -> str:
    benchmark_name = "sparse_attention"
    bandwidths = [1023, 512, 256, 128, 64, 32, 16, 0]

    configs = [
        ("sparse_attention_baseline.mlir", "S"),
        ("sparse_attention_baseline.mlir", "SL"),
        ("sparse_attention_dense.mlir", "ARL"),
        ("sparse_attention_dia.mlir", "ARL"),
        ("sparse_attention_dia.mlir", "DRL"),
        ("sparse_attention_dia_inputs.mlir", "ARL"),
        ("sparse_attention_dia_inputs.mlir", "DRL"),
    ]

    result_path = bench.run(
        benchmark_name=benchmark_name,
        program_dir="./benchmarking/programs",
        configs=configs,
        bandwidths=bandwidths,
        warmup=1,
        runs_count=runs_count,
    )

    return result_path


def chained_matmul(runs_count: int = 5) -> str:
    benchmark_name = "chain"
    bandwidths = [1023, 512, 256, 128, 64, 32, 16, 0]

    configs = [
        ("chain10_dense.mlir", None),
        ("chain10_dense.mlir", "L"),
        ("chain10_dense.mlir", "ARL"),
        ("chain10_dia.mlir", "ARL"),
        ("chain10_dia.mlir", "DRL"),
        ("chain10_dia_inputs.mlir", "ARL"),
        ("chain10_dia_inputs.mlir", "DRL"),
    ]

    result_path = bench.run(
        benchmark_name=benchmark_name,
        program_dir="./benchmarking/programs",
        configs=configs,
        bandwidths=bandwidths,
        warmup=1,
        runs_count=runs_count,
    )

    return result_path


def attention_propagation(runs_count: int = 5):
    from pathlib import Path

    RESULT_DIR = Path("./results")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_DIR / "attention_sparsity.csv"
    csv_path.write_text("")

    hw = False
    for lb in [3, 256, 512, 1023]:
        hw = attention_prop.run(
            N=1024, lb=lb, ub=0, csv_path=csv_path, header_written=hw
        )

    return csv_path


def comptime_experiment(runs_count: int = 5) -> str:
    benchmark_name = "compilation_time"
    configs = [
        ("analysis_rewrite", ["--banded-analysis --banded-rewrite"]),
    ]
    result_path = comptime.run_experiment(
        benchmark_name,
        configs,
        sizes=[128, 256, 512, 1024, 2048],
        ops_range=[20, 30, 50, 100, 200],
    )

    return result_path


def stur_chain() -> str:
    stur_dir = os.environ.get("STUR_DIR", "/app/stur")
    subprocess.run(["python3", "benchmark_chained.py"], cwd=stur_dir, check=True)

    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    dest = results_dir / "symbolic_chained_stur.csv"
    shutil.copy(Path(stur_dir) / "benchmark_chained_results.csv", dest)
    return str(dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Comma-separated list of benchmarks",
        default="kalman_filter,sparse_attention,batch_bertlike,chain,attention_prop,comptime,bench_chain",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        help="Number of executions for each benchmark",
        default=10,
    )

    args = parser.parse_args()

    dispatch = {
        "kalman_filter": kalman_filter,
        "batch_bertlike": batch_bertlike,
        "chain": chained_matmul,
        "sparse_attention": sparse_attention,
        "attention_prop": attention_propagation,
        "comptime": comptime_experiment,
        "bench_chain": stur_chain,
    }

    benchmarks = [x.strip() for x in args.benchmark.split(",")]

    benchmark_files = {}
    for benchmark in benchmarks:
        benchmark_func = dispatch[benchmark]
        try:
            if "runs_count" in inspect.signature(benchmark_func).parameters:
                result_path = benchmark_func(args.repeat)
            else:
                result_path = benchmark_func()
            benchmark_files[benchmark] = result_path
        except KeyError as e:
            print(f"error: invalid benchmark option {str(e)}")

    runtime_benchmarks = {
        "kalman_filter",
        "sparse_attention",
        "batch_bertlike",
        "chain",
    }

    if runtime_benchmarks.issubset(benchmark_files.keys()):
        runtime_csvs = [
            benchmark_files["kalman_filter"],
            benchmark_files["sparse_attention"],
            benchmark_files["batch_bertlike"],
            benchmark_files["chain"],
        ]

        print("Building Figure 11...")
        figures.figure11(runtime_csvs)

        print("Building Figure 12...")
        figures.figure12(runtime_csvs)

    if "bench_chain" in benchmark_files:
        print("Building Figure 10...")
        figures.figure10()

    if "comptime" in benchmark_files:
        comptime_csv = benchmark_files["comptime"]
        attention_prop_csv = benchmark_files["attention_prop"]

        print("Building Figure 13...")
        figures.figure13(
            csv_path=comptime_csv,
        )

        print("Building Figure 14...")
        figures.figure14(
            csv_path=comptime_csv,
        )

        print("Building Figure 15...")
        figures.figure15(
            csv_path=attention_prop_csv,
        )
