import re
from pathlib import Path
import subprocess
import csv
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

BUILD_DIR  = Path("./build")
TMP_DIR    = Path("/tmp/bpa_main")
RESULT_DIR = Path("./results")

TENSORS = ["Q", "K", "Mask", "QK", "MaskedQK", "AttnWeights", "V", "Output"]
COMPUTED_POS = {"QK": "0", "MaskedQK": "1", "AttnWeights": "2", "Output": "3"}

_MD_RE = re.compile(
    r"%(\w+)\s*=.*?"
    r"(?:lowerBw\s*=\s*(\d+)\s*:\s*i64.*?upperBw\s*=\s*(\d+)\s*:\s*i64"
    r"|upperBw\s*=\s*(\d+)\s*:\s*i64.*?lowerBw\s*=\s*(\d+)\s*:\s*i64)",
    re.DOTALL,
)


def run_cmd(cmd, input_text=None):
    result = subprocess.run(cmd, input=input_text, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed: {' '.join(cmd)}")
        logger.error(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return result.stdout


def substitute(src: Path, dst: Path, lb: int, ub: int, N: int):
    dst.write_text(src.read_text().replace("X", str(lb)).replace("Y", str(ub)).replace("Z", str(N)))


def parse_bws(ir: str, N) -> dict:
    result = {}
    for m in _MD_RE.finditer(ir):
        name = m.group(1)
        result[name] = (
            (min(N - 1, int(m.group(2))), min(N - 1, int(m.group(3)))) if m.group(2) is not None
            else (min(N - 1, int(m.group(5))), min(N - 1, int(m.group(4))))
        )
    return result


def get_bws(ir: str, N: int, lb: int, ub: int) -> dict:
    parsed = parse_bws(ir, N)
    full = (N - 1, N - 1)
    return {
        "Q":           full,
        "K":           full,
        "V":           full,
        "Mask":        (lb, ub),
        "QK":          parsed.get(COMPUTED_POS["QK"],          full),
        "MaskedQK":    parsed.get(COMPUTED_POS["MaskedQK"],    full),
        "AttnWeights": parsed.get(COMPUTED_POS["AttnWeights"], full),
        "Output":      parsed.get(COMPUTED_POS["Output"],      full),
    }


def nnz(N: int, lo: int, up: int) -> int:
    lo, up = min(lo, N - 1), min(up, N - 1)
    return sum(min(i + up, N - 1) - max(i - lo, 0) + 1 for i in range(N))


def sparsity(N: int, lo: int, up: int) -> float:
    return 1.0 - nnz(N, lo, up) / (N * N)


def write_csv_row(csv_path: Path, row: dict, write_header: bool = False):
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
        f.flush()


def run(N: int, lb: int, ub: int, csv_path: Path, header_written: bool) -> bool:
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    src       = Path("benchmarking/programs/sparse_attention_prop.mlir")
    mlir_file = TMP_DIR / f"attention_{N}_{lb}_{ub}.mlir"

    substitute(src, mlir_file, lb, ub, N)
    alg_opt = f"{BUILD_DIR}/tools/alg-opt"

    stages = {
        "initial":  get_bws(mlir_file.read_text(), N, lb, ub),
        "forward":  get_bws(run_cmd([alg_opt, str(mlir_file), "--banded-analysis=disable-bw=true"]), N, lb, ub),
        "backward": get_bws(run_cmd([alg_opt, str(mlir_file), "--banded-analysis"]), N, lb, ub),
    }

    for stage, bws in stages.items():
        for tensor in TENSORS:
            lo, up = bws[tensor]
            row = {
                "N": N, "lowerBw": lb, "upperBw": ub,
                "stage": stage, "tensor": tensor,
                "tensor_lowerBw": lo, "tensor_upperBw": up,
                "nnz": nnz(N, lo, up),
                "sparsity": f"{sparsity(N, lo, up):.6f}",
            }
            write_csv_row(csv_path, row, write_header=not header_written)
            header_written = True

    logger.info(f"N={N} lb={lb} ub={ub} done")
    return header_written


if __name__ == "__main__":
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULT_DIR / "attention_sparsity.csv"
    csv_path.write_text("")

    hw = False
    for lb in [3, 256, 512, 1023]:
        hw = run(N=1024, lb=lb, ub=0, csv_path=csv_path, header_written=hw)

    logger.info(f"Results saved to {csv_path}")
