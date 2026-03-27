import bench


def chain_matmul():
    bandwidths = [512, 409, 307, 256, 204, 153, 102, 51, 0]

    configs = [
        ("chain_matmul_dense.mlir", None),
        ("chain_matmul_dense.mlir", "analysis", "rewrite"),
        ("chain_matmul_dia.mlir", "analysis", "rewrite"),
        ("chain_matmul_dia.mlir", "analysis-detect", "rewrite"),
    ]

    bench.run(
        benchmark_name="chain_matmul",
        program_dir="./benchmarking/programs",
        configs=configs,
        bandwidths=bandwidths,
        warmup=1,
        runs_count=5,
    )


def kalman_filter():
    bandwidths = [512, 409, 307, 256, 204, 153, 102, 51, 0]

    configs = [
        ("kalman_filter_dense.mlir", None),
        ("kalman_filter_dense.mlir", "analysis", "rewrite"),
        ("kalman_filter_dia.mlir", "analysis", "rewrite"),
        ("kalman_filter_dia.mlir", "analysis-detect", "rewrite"),
    ]

    bench.run(
        benchmark_name="kalman_filter",
        program_dir="./benchmarking/programs",
        configs=configs,
        bandwidths=bandwidths,
        warmup=1,
        runs_count=5,
    )


if __name__ == "__main__":
    chain_matmul()
    kalman_filter()
