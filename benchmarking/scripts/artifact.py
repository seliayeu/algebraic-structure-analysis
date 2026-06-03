import argparse
import figures
import comptime
import bench


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Comma-separated list of benchmarks",
        default="kalman_filter,sparse_attention,batch_bertlike,chain,comptime",
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
        "comptime": comptime_experiment,
    }

    benchmarks = [x.strip() for x in args.benchmark.split(",")]

    benchmark_files = {}
    for benchmark in benchmarks:
        benchmark_func = dispatch[benchmark]
        try:
            result_path = benchmark_func(args.repeat)
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

    if "comptime" in benchmark_files:
        comptime_csv = benchmark_files["comptime"]

        print("Building Figure 14...")
        figures.figure14(
            csv_path=comptime_csv,
        )

        print("Building Figure 15...")
        figures.figure15(
            csv_path=comptime_csv,
        )
