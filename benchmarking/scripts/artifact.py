import argparse
import figures
import bench


def kalman_filter(runs_count: int = 5):
    benchmark_name = "kalman_filter"
    bandwidths = [512, 409, 307, 256, 204, 153, 102, 51, 0]

    configs = [
        ("kalman_filter_dense.mlir", None),
        ("kalman_filter_dense.mlir", "analysis", "rewrite"),
        ("kalman_filter_dia.mlir", "analysis", "rewrite"),
        ("kalman_filter_dia.mlir", "analysis-detect", "rewrite"),
    ]

    color_palette = [
        "#94A3B8",  # gray
        "#F97316",  # orange
        "#06B6D4",  # cyan
        "#00b200",  # green
    ]

    result_path = bench.run(
        benchmark_name=benchmark_name,
        program_dir="./benchmarking/programs",
        configs=configs,
        bandwidths=bandwidths,
        warmup=1,
        runs_count=runs_count,
    )

    result_path = "./results/kalman_filter.csv"
    figures.bandwidth_plot(
        result_path,
        benchmark_name,
        "Kalman Filter Benchmark",
        color_palette=color_palette,
    )


def batch_bertlike(runs_count: int = 5):
    benchmark_name = "batch_bertlike"
    bandwidths = [512, 409, 307, 256, 204, 153, 102, 51, 0]

    configs = [
        ("batch_bertlike_dense.mlir", None),
        ("batch_bertlike_dense.mlir", "analysis", "rewrite"),
        ("batch_bertlike_dia.mlir", "analysis", "rewrite"),
        ("batch_bertlike_dia.mlir", "analysis-detect", "rewrite"),
        ("batch_bertlike_dia_inputs.mlir", "analysis", "rewrite"),
        ("batch_bertlike_dia_inputs.mlir", "analysis-detect", "rewrite"),
    ]

    color_palette = [
        "#94A3B8",  # gray
        "#8B5CF6",  # Vibrant Purple
        "#EC4899",  # Hot Pink
        "#3B82F6",  # Bright Blue
        "#F97316",  # Orange
        "#06B6D4",  # Cyan
    ]

    result_path = bench.run(
        benchmark_name=benchmark_name,
        program_dir="./benchmarking/programs",
        configs=configs,
        bandwidths=bandwidths,
        warmup=1,
        runs_count=runs_count,
    )

    figures.bandwidth_plot(
        result_path,
        benchmark_name,
        "Batch Bert-Like Benchmark",
        color_palette=color_palette,
    )

# def chained_matmul(runs_count: int = 5):
#     benchmark_name = "chain100"
#     bandwidths = [512, 409, 307, 256, 204, 153, 102, 51, 0]
#
#     configs = [
#         ("chain100_dense.mlir", None),
#         ("chain100_dense.mlir", "analysis", "rewrite"),
#         ("chain100_dia.mlir", "analysis", "rewrite"),
#         ("chain100_dia.mlir", "analysis-detect", "rewrite"),
#         ("chain100_dia_inputs.mlir", "analysis", "rewrite"),
#         ("chain100_dia_inputs.mlir", "analysis-detect", "rewrite"),
#     ]
#
#     color_palette = [
#         "#94A3B8",  # gray
#         "#8B5CF6",  # Vibrant Purple
#         "#EC4899",  # Hot Pink
#         "#3B82F6",  # Bright Blue
#         "#F97316",  # Orange
#         "#06B6D4",  # Cyan
#     ]
#
#     result_path = bench.run(
#         benchmark_name=benchmark_name,
#         program_dir="./benchmarking/programs",
#         configs=configs,
#         bandwidths=bandwidths,
#         warmup=1,
#         runs_count=runs_count,
#     )
#
#     figures.bandwidth_plot(
#         result_path,
#         benchmark_name,
#         "Batch Bert-Like Benchmark",
#         color_palette=color_palette,
#     )


if __name__ == "__main__":
    dispatch = {
        "kalman_filter": kalman_filter,
        "batch_bertlike": batch_bertlike,
        "chain100": chained_matmul,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=str,
        help="Comma-separated list of figures",
        # default="7,8,9,10,11,12",
        default="kalman_filter,batch_bertlike",
    )

    parser.add_argument(
        "--repeat",
        type=int,
        help="Number of executions for each benchmark",
        default=5,
    )

    args = parser.parse_args()

    benchmarks = [x.strip() for x in args.benchmark.split(",")]
    for benchmark in benchmarks:
        try:
            func = dispatch[benchmark]
            func(args.repeat)
        except KeyError as e:
            print(f"error: invalid benchmark option {str(e)}")
