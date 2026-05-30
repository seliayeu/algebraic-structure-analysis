import argparse
import figures
import comptime
import bench


def kalman_filter(runs_count: int = 5):
    benchmark_name = "kalman_filter"
    bandwidths = [1023, 512, 409, 307, 256, 204, 153, 102, 51, 0]
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


def batch_bertlike(runs_count: int = 5, plot=True):
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


def sparse_attention(runs_count: int = 5):
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


def chained_matmul(runs_count: int = 5):
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


def comptime_experiment(runs_count: int = 5):
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

    figures.create_comptime_plot(
        csv_path=result_path,
        output_prefix="bpa_comptime",
        figure_title="Compilation Time Breakdown by Component",
        show_title=True,
        save_png_and_svg=True,
    )


if __name__ == "__main__":
    dispatch = {
        "kalman_filter": kalman_filter,
        "batch_bertlike": batch_bertlike,
        "chain100": chained_matmul,
        "sparse_attention": sparse_attention,
        "comptime": comptime_experiment,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Comma-separated list of figures",
        # default="7,8,9,10,11,12",
        default="kalman_filter,sparse_attention,batch_bertlike,chain100,comptime",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        help="Number of executions for each benchmark",
        default=10,
    )

    args = parser.parse_args()

    benchmarks = [x.strip() for x in args.benchmark.split(",")]
    for benchmark in benchmarks:
        try:
            func = dispatch[benchmark]
            func(args.repeat)
        except KeyError as e:
            print(f"error: invalid benchmark option {str(e)}")
