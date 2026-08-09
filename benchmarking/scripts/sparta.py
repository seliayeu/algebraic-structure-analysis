import csv
import os
import re
import statistics
import subprocess
from pathlib import Path


KS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
SIZE = 2048


def _sparta_dir() -> Path:
    return Path(os.environ.get("SPARTA_DIR", "/opt/sparta"))


def _binary_path(k: int) -> Path:
    sparta_dir = _sparta_dir()
    return sparta_dir / "bin" / f"tesa-prop-{k}"


def _ensure_binary(k: int) -> Path:
    binary = _binary_path(k)
    if binary.exists():
        return binary

    sparta_dir = _sparta_dir()
    if not (sparta_dir / "CMakeLists.txt").exists():
        raise FileNotFoundError(
            f"SPARTA binary {binary} not found and {sparta_dir} is not a source tree"
        )

    build_root = Path(os.environ.get("SPARTA_BUILD_ROOT", "build/sparta"))
    build_dir = build_root / f"k{k}"
    build_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "cmake",
            "-G",
            "Ninja",
            "-S",
            str(sparta_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DSIZE_MACRO={SIZE}",
            f"-DNUM_MATMULS={k}",
        ],
        check=True,
    )
    subprocess.run(["cmake", "--build", str(build_dir), "--target", "tesa-prop"], check=True)
    return build_dir / "tesa-prop"


def _run_once(binary: Path) -> float:
    output = subprocess.check_output([str(binary)], stderr=subprocess.STDOUT).decode()
    match = re.search(r"Took\s+([\d.eE+-]+)\s+s", output)
    if not match:
        raise RuntimeError(f"Could not parse SPARTA timing from output:\n{output}")
    return float(match.group(1)) * 1000.0


def run_benchmark(repetitions: int = 10) -> str:
    repetitions = max(1, int(repetitions))

    results_dir = Path("./results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = results_dir / "symbolic_chained_tesa.csv"

    print("\nRunning benchmark for mode: SPARTA/TESA")
    print(f"{'K':<6} | {'Avg Time (ms)':<15} | {'Trials (ms)':<30}")
    print("-" * 60)

    with csv_filename.open("w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["k", "avg_time_ms"] + [f"trial_{i + 1}" for i in range(repetitions)])

        for k in KS:
            binary = _ensure_binary(k)

            # Warmup run; exclude it from the CSV just like the BPA benchmark.
            _run_once(binary)

            trials = [_run_once(binary) for _ in range(repetitions)]
            avg_time = statistics.mean(trials)
            trials_str = ", ".join(f"{t:.2f}" for t in trials)
            print(f"{k:<6} | {avg_time:<15.2f} | {trials_str}")

            csv_writer.writerow([k, avg_time] + trials)
            csvfile.flush()

    print(f"Results saved to {csv_filename}")
    return str(csv_filename)
