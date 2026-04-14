import platform
from pathlib import Path
import subprocess
import csv
import logging


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)

PROGRAM_DIR = Path("./benchmarking/scripts/")
RESULT_DIR = Path("./results")
RESULT_DIR.mkdir(exist_ok=True)


# -----------------------------
# Run with /usr/bin/time -v
# -----------------------------
def run_benchmark_full(bw: int):
    system = platform.system()

    if system == "Darwin":
        cmd = [
            "/usr/bin/time",
            "-l",
            "/Users/kaio/projects/algebraic-structure-analysis/venv/bin/python3",
            str(PROGRAM_DIR) + "/scipy_kernel.py",
            str(bw),
        ]
    else:
        cmd = [
            "/usr/bin/time",
            "-l",
            "/Users/kaio/projects/algebraic-structure-analysis/venv/bin/python3",
            str(PROGRAM_DIR) + "/scipy_kernel.py",
            str(bw),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(result.stderr)
        raise RuntimeError(f"Benchmark failed: {result.stderr}")

    out_lines = result.stdout.splitlines()
    err_lines = result.stderr.splitlines()

    # -----------------------------
    # Parse program output
    # -----------------------------
    total = float(
        next(l for l in out_lines if "total time:" in l).split(":")[1].split()[0]
    )
    avg = float(next(l for l in out_lines if "avg time:" in l).split(":")[1].split()[0])

    stats = {
        "bw": bw,
        "total_time_s": total,
        "avg_time_s": avg,
    }

    # -----------------------------
    # Parse /usr/bin/time -v output
    # -----------------------------
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
                stats["max_rss_kb"] = int(line.split()[-1])
                stats["max_rss_mb"] = stats["max_rss_kb"] / 1024
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

    return stats


# -----------------------------
# CSV writer (same as yours)
# -----------------------------
def write_csv_row(csv_path: Path, row_dict: dict, write_header: bool = False):
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)
        f.flush()


# -----------------------------
# Main benchmark loop
# -----------------------------
def run():
    bandwidths = [512, 409, 307, 256, 204, 153, 102, 51, 0]

    csv_path = RESULT_DIR / "scipy_dia_benchmark.csv"
    csv_path.write_text("")
    header_written = False

    total_configs = len(bandwidths)
    current = 0

    logger.info("=" * 80)
    logger.info("STARTING SCIPY DIA BENCHMARK")
    logger.info("=" * 80)

    for bw in bandwidths:
        current += 1
        logger.info(f"[{current}/{total_configs}] bandwidth={bw}")

        stats = run_benchmark_full(bw)

        write_csv_row(csv_path, stats, write_header=not header_written)
        header_written = True

        logger.info(
            f"[{current}/{total_configs}] done bw={bw} "
            f"avg={stats['avg_time_s']:.6f}s "
            f"rss={stats.get('max_rss_mb', 0):.2f} MB"
        )

    logger.info("=" * 80)
    logger.info(f"RESULTS SAVED TO {csv_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    run()
