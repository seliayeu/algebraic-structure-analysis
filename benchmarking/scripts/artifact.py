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
import symbolic
import sparta


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


def attention_backward_prop(runs_count: int = 5) -> str:
    benchmark_name = "attention_backward_prop"
    # lowerBw sweep matches Figure 15; upperBw is fixed at 0 in the template
    bandwidths = [3, 256, 512, 1023]

    configs = [
        ("sparse_attention_baseline.mlir", "S"),  # naive baseline (no analysis)
        ("sparse_attention_dense_asym.mlir", "FRL"),  # forward-only analysis + rewrite
        ("sparse_attention_dense_asym.mlir", "ARL"),  # full forward+backward + rewrite
    ]

    return bench.run(
        benchmark_name=benchmark_name,
        program_dir="./benchmarking/programs",
        configs=configs,
        bandwidths=bandwidths,
        warmup=1,
        runs_count=runs_count,
    )


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


def symbolic_chain(runs_count: int = 10) -> list[str]:
    return symbolic.run_benchmark(repetitions=runs_count)


def stur_chain() -> str:
    stur_dir = os.environ.get("STUR_DIR", "/app/stur")
    subprocess.run(["python3", "benchmark_chained.py"], cwd=stur_dir, check=True)

    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    dest = results_dir / "symbolic_chained_stur.csv"
    shutil.copy(Path(stur_dir) / "benchmark_chained_results.csv", dest)
    return str(dest)


def sparta_chain(runs_count: int = 10) -> str:
    return sparta.run_benchmark(repetitions=runs_count)


FIGURE_REQUIREMENTS = {
    "figure10": ["symbolic_chain", "bench_chain", "sparta_chain"],
    "figure11": ["kalman_filter", "sparse_attention", "batch_bertlike", "chain"],
    "figure12": ["kalman_filter", "sparse_attention", "batch_bertlike", "chain"],
    "figure13": ["comptime"],
    "figure14": ["comptime"],
    "figure15": ["attention_prop"],
    "figure16": ["attention_backward_prop"],
}

BENCHMARK_DISPATCH = {
    "kalman_filter": kalman_filter,
    "batch_bertlike": batch_bertlike,
    "chain": chained_matmul,
    "sparse_attention": sparse_attention,
    "attention_prop": attention_propagation,
    "attention_backward_prop": attention_backward_prop,
    "comptime": comptime_experiment,
    "symbolic_chain": symbolic_chain,
    "bench_chain": stur_chain,
    "sparta_chain": sparta_chain,
}


def run_required_benchmarks(figures, runs):
    required_benchmarks = set()
    for fig in figures:
        if fig in FIGURE_REQUIREMENTS:
            required_benchmarks.update(FIGURE_REQUIREMENTS[fig])
        else:
            print(f"Warning: unknown figure '{fig}' – skipping.")

    benchmark_results = {}
    for bench_name in required_benchmarks:
        func = BENCHMARK_DISPATCH.get(bench_name)
        if func is None:
            print(f"Error: no benchmark function for '{bench_name}'")
            continue
        try:
            if "runs_count" in inspect.signature(func).parameters:
                result_path = func(runs)
            else:
                result_path = func()
            benchmark_results[bench_name] = result_path
            print(f"Finished benchmark '{bench_name}' -> {result_path}")
        except Exception as e:
            print(f"Error running benchmark '{bench_name}': {e}")

    return benchmark_results


def generate_requested_figures(figures_list, benchmark_results):
    def get_runtime_csvs():
        return [
            benchmark_results["kalman_filter"],
            benchmark_results["sparse_attention"],
            benchmark_results["batch_bertlike"],
            benchmark_results["chain"],
        ]

    for fig in figures_list:
        if fig == "figure10":
            if all(b in benchmark_results for b in FIGURE_REQUIREMENTS["figure10"]):
                print("Building Figure 10...")
                figures.figure10()
            else:
                print("Skipping Figure 10: missing symbolic chain benchmarks.")

        elif fig == "figure11":
            if all(b in benchmark_results for b in FIGURE_REQUIREMENTS["figure11"]):
                print("Building Figure 11...")
                figures.figure11(get_runtime_csvs())
            else:
                print("Skipping Figure 11: missing runtime benchmarks.")

        elif fig == "figure12":
            if all(b in benchmark_results for b in FIGURE_REQUIREMENTS["figure12"]):
                print("Building Figure 12...")
                figures.figure12(get_runtime_csvs())
            else:
                print("Skipping Figure 12: missing runtime benchmarks.")

        elif fig == "figure13":
            if "comptime" in benchmark_results:
                print("Building Figure 13...")
                figures.figure13(csv_path=benchmark_results["comptime"])
            else:
                print(
                    "Skipping Figure 13: required benchmark 'comptime' not available."
                )

        elif fig == "figure14":
            if "comptime" in benchmark_results:
                print("Building Figure 14...")
                figures.figure14(csv_path=benchmark_results["comptime"])
            else:
                print(
                    "Skipping Figure 14: required benchmark 'comptime' not available."
                )

        elif fig == "figure15":
            if "attention_prop" in benchmark_results:
                print("Building Figure 15...")
                figures.figure15(csv_path=benchmark_results["attention_prop"])
            else:
                print(
                    "Skipping Figure 15: required benchmark 'attention_prop' not available."
                )

        elif fig == "figure16":
            if "attention_backward_prop" in benchmark_results:
                print("Building Figure 16...")
                figures.figure16(csv_path=benchmark_results["attention_backward_prop"])
            else:
                print(
                    "Skipping Figure 16: required benchmark 'attention_backward_prop' not available."
                )

        else:
            print(f"Figure '{fig}' is not recognised – nothing generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--figures",
        type=str,
        help="Comma-separated list of figure numbers to generate (e.g., 10,11)",
        default="10,11,12,13,14,15,16",
    )
    parser.add_argument(
        "--runs",
        type=int,
        help="Number of executions for each benchmark",
        default=10,
    )
    args = parser.parse_args()

    raw = [f.strip() for f in args.figures.split(",")]
    requested_figures = []
    for item in raw:
        if item.startswith("figure"):
            num = item[6:]
        else:
            num = item
        requested_figures.append(f"figure{num}")

    results = run_required_benchmarks(requested_figures, args.runs)
    generate_requested_figures(requested_figures, results)
