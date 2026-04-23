from pathlib import Path
from enum import Enum, auto

N = 1024
NUM_MATMULS = 10

class Mode(Enum):
    DENSE      = auto()
    DIA        = auto()
    DIA_INPUTS = auto()


# ── Type/metadata helpers ─────────────────────────────────────────────────────

def dense_type() -> str:
    return f"tensor<{N}x{N}xf32>"

def dia_static_type() -> str:
    return f"tensor<Yx{N}xf32>"

def dia_dyn_type() -> str:
    return f"tensor<?x{N}xf32>"

def dense_meta() -> str:
    return "{metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}}"

def dia_meta() -> str:
    return "{metadata = {dia = true, lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}}"

def dia_inputs_meta() -> str:
    return "{metadata = {lowerBw = X : i64, upperBw = X : i64, propertyDims = [0, 1]}}"


# ── Generator ─────────────────────────────────────────────────────────────────

def generate_chain_matmul(n: int, mode: Mode, out_path: Path):
    lines = []
    lines.append("func.func @kernel() -> f32 {")

    # ── empty tensors ─────────────────────────────────────────────────────────
    if mode == Mode.DIA:
        dyn_t = dia_dyn_type()
        lines.append("  %c0 = arith.constant 0 : index")
        for i in range(1, n + 2):
            lines.append(f"  %e{i} = tensor.empty(%c0) : {dyn_t}")
    else:
        t = dense_type()
        for i in range(1, n + 2):
            lines.append(f"  %e{i} = tensor.empty() : {t}")

    lines.append("")

    # ── constants ─────────────────────────────────────────────────────────────
    if mode == Mode.DENSE:
        t, meta = dense_type(), dense_meta()
    elif mode == Mode.DIA:
        t, meta = dia_static_type(), dia_meta()
    else:  # DIA_INPUTS
        t, meta = dense_type(), dia_inputs_meta()

    lines.append(f"  %input = arith.constant {meta} dense<1.0> : {t}")
    for i in range(1, n + 1):
        lines.append(f"  %W{i} = arith.constant {meta} dense<1.0> : {t}")

    lines.append("")

    # ── matmul chain ──────────────────────────────────────────────────────────
    if mode == Mode.DENSE:
        op, static_t, inter_t = "linalg.matmul", dense_type(), dense_type()
    elif mode == Mode.DIA:
        op, static_t, inter_t = "dia.matmul", dia_static_type(), dia_dyn_type()
    else:  # DIA_INPUTS
        op, static_t, inter_t = "dia.matmul", dense_type(), dense_type()

    lines.append(
        f"  %m1 = {op} ins(%input, %W1 : {static_t}, {static_t})"
        f" outs(%e1 : {inter_t}) -> {inter_t}"
    )
    for i in range(2, n + 1):
        lines.append(
            f"  %m{i} = {op} ins(%m{i-1}, %W{i} : {inter_t}, {static_t})"
            f" outs(%e{i} : {inter_t}) -> {inter_t}"
        )

    lines.append("")

    # ── residual add ──────────────────────────────────────────────────────────
    if mode == Mode.DENSE:
        add_op = "linalg.add"
    else:
        add_op = "dia.elementwise kind = <add>"

    lines.append(
        f"  %result_add = {add_op}"
        f" ins(%m{n}, %input : {inter_t}, {static_t})"
        f" outs(%e{n+1} : {inter_t}) -> {inter_t}"
    )

    lines.append("")

    # ── extract scalar ────────────────────────────────────────────────────────
    idx = "0" if mode == Mode.DIA else "5"
    lines.append(f"  %index = arith.constant {idx} : index")
    lines.append(f"  %val = tensor.extract %result_add[%index, %index] : {inter_t}")
    lines.append("  return %val : f32")
    lines.append("}")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Written: {out_path}  ({n} matmuls, {mode.name})")


if __name__ == "__main__":
    out_dir = Path("./benchmarking/programs")
    out_dir.mkdir(parents=True, exist_ok=True)

    generate_chain_matmul(NUM_MATMULS, Mode.DENSE,      out_dir / f"chain{NUM_MATMULS}_dense.mlir")
    generate_chain_matmul(NUM_MATMULS, Mode.DIA,        out_dir / f"chain{NUM_MATMULS}_dia.mlir")
    generate_chain_matmul(NUM_MATMULS, Mode.DIA_INPUTS, out_dir / f"chain{NUM_MATMULS}_dia_inputs.mlir")
