import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


COMBO_STYLES = {
    "dia_dia_dia":       {"color": "#06B6D4", "label": "dia × dia → dia"},
    "dia_dia_dense":     {"color": "#EC4899", "label": "dia × dia → dense"},
    "dia_dense_dia":     {"color": "#F97316", "label": "dia × dense → dia"},
    "dia_dense_dense":   {"color": "#EF4444", "label": "dia × dense → dense"},
    "dense_dia_dense":   {"color": "#00b200", "label": "dense × dia → dense"},
    "dense_dense_dia":   {"color": "#8B5CF6", "label": "dense × dense → dia"},
    "dense_dense_dense": {"color": "#94A3B8", "label": "dense × dense → dense"},
}


def plot_matmul_combinations(
    csv_path: str,
    out_path: str = "results/matmul_combinations.png",
    title: str = "Matmul Combination Benchmark",
):
    df = pd.read_csv(csv_path)
    df["combo"] = df["lhs"] + "_" + df["rhs"] + "_" + df["out"]
    df = df.sort_values("bw")

    fig, ax = plt.subplots(figsize=(10, 6))

    for combo, style in COMBO_STYLES.items():
        subset = df[df["combo"] == combo]
        if subset.empty:
            continue
        ax.plot(
            subset["bw"],
            subset["avg_time_s"],
            marker="o",
            linewidth=2,
            markersize=5,
            color=style["color"],
            label=style["label"],
        )

    ax.set_xlabel("Bandwidth", fontsize=12)
    ax.set_ylabel("Avg Time (s)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    plot_matmul_combinations(
        csv_path="results/matmul_combinations.csv",
        out_path="results/matmul_combinations.png",
    )
