import argparse
import figures
import bench


def chain_matmul(runs_count: int = 5):
    benchmark_name = "chain_matmul"
    bandwidths = [512, 409, 307, 256, 204, 153, 102, 51, 0]

    configs = [
        ("chain_matmul_dense.mlir", None),
        ("chain_matmul_dense.mlir", "analysis", "rewrite"),
        ("chain_matmul_dia.mlir", "analysis", "rewrite"),
        ("chain_matmul_dia.mlir", "analysis-detect", "rewrite"),
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
        result_path, benchmark_name, "Chain Matrix Multiply Benchmark"
    )


def kalman_filter(runs_count: int = 5):
    benchmark_name = "kalman_filter"
    bandwidths = [512, 409, 307, 256, 204, 153, 102, 51, 0]

    configs = [
        ("kalman_filter_dense.mlir", None),
        ("kalman_filter_dense.mlir", "analysis", "rewrite"),
        ("kalman_filter_dia.mlir", "analysis", "rewrite"),
        ("kalman_filter_dia.mlir", "analysis-detect", "rewrite"),
    ]

    result_path = bench.run(
        benchmark_name=benchmark_name,
        program_dir="./benchmarking/programs",
        configs=configs,
        bandwidths=bandwidths,
        warmup=1,
        runs_count=runs_count,
    )
    figures.bandwidth_plot(result_path, benchmark_name, "Kalman Filter Benchmark")


def batch_bertlike(runs_count: int = 5):
    benchmark_name = "batch_bertlike"
    bandwidths = [512, 409, 307, 256, 204, 153, 102, 51, 0]

    configs = [
        ("batch_bertlike_dense.mlir", None),
        ("batch_bertlike_dense.mlir", "analysis", "rewrite"),
        ("batch_bertlike_dia.mlir", "analysis", "rewrite"),
        ("batch_bertlike_dia.mlir", "analysis-detect", "rewrite"),
    ]

    result_path = bench.run(
        benchmark_name=benchmark_name,
        program_dir="./benchmarking/programs",
        configs=configs,
        bandwidths=bandwidths,
        warmup=1,
        runs_count=runs_count,
    )
    figures.bandwidth_plot(result_path, benchmark_name, "Batch Bert-Like Benchmark")


if __name__ == "__main__":
    dispatch = {
        "chain": chain_matmul,
        "kalman_filter": kalman_filter,
        "batch_bertlike": batch_bertlike,
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
