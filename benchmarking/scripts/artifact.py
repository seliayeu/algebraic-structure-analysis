import figures
import bench


def chain_matmul():
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
        runs_count=5,
    )
    figures.bandwidth_plot(
        result_path, benchmark_name, "Chain Matrix Multiply Benchmark"
    )


def kalman_filter():
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
        runs_count=5,
    )
    figures.bandwidth_plot(result_path, benchmark_name, "Kalman Filter Benchmark")


def batch_bertlike():
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
        runs_count=5,
    )
    figures.bandwidth_plot(result_path, benchmark_name, "Batch Bert-Like Benchmark")


if __name__ == "__main__":
    chain_matmul()
    kalman_filter()
    batch_bertlike()
