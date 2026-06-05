import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

RESULT_DIR = Path("./results")

STAGE_COLORS = {
    "initial":  "#aaaaaa",
    "forward":  "#e99575",
    "backward": "#71b5a0",
}
STAGES = ["initial", "forward", "backward"]


def create_plot(
    csv_path=RESULT_DIR / "attention_sparsity.csv",
    output_prefix="sparsity",
    show_title: bool = False,
    save_png_and_svg: bool = True,
):
    df = pd.read_csv(csv_path)

    agg = (
        df.groupby(["N", "lowerBw", "upperBw", "stage"])
        .agg(total_nnz=("nnz", "sum"), n_tensors=("tensor", "count"))
        .reset_index()
    )
    agg["total_sparsity"] = 1.0 - agg["total_nnz"] / (agg["n_tensors"] * agg["N"] ** 2)

    def mask_label(row):
        lb, ub, N = row["lowerBw"], row["upperBw"], row["N"]
        return f"Causal<br>(LBW={lb}, UBW={ub})" if lb == N - 1 else f"Window<br>(LBW={lb}, UBW={ub})"

    agg["label"] = agg.apply(mask_label, axis=1)

    # Sort groups by lowerBw so they appear in ascending order
    mask_order = (
        agg[["label", "lowerBw"]].drop_duplicates()
        .sort_values("lowerBw")["label"]
        .tolist()
    )

    n_masks  = len(mask_order)
    n_stages = len(STAGES)
    bar_width = 0.8 / n_stages

    fig = go.Figure()

    for mask_idx, label in enumerate(mask_order):
        subset = agg[agg["label"] == label]
        for stage_idx, stage in enumerate(STAGES):
            row = subset[subset["stage"] == stage]
            if row.empty:
                continue
            val = row["total_sparsity"].iloc[0]
            x   = mask_idx + (stage_idx - n_stages / 2 + 0.5) * bar_width

            fig.add_trace(go.Bar(
                name=stage.capitalize(),
                x=[x],
                y=[val],
                marker_color=STAGE_COLORS[stage],
                marker_line_color="black",
                marker_line_width=0.8,
                opacity=0.9,
                width=bar_width * 0.85,
                showlegend=(mask_idx == 0),
                legendgroup=stage,
                hovertemplate=(
                    f"<b>{label.replace('<br>', ' ')}</b><br>"
                    f"Stage: {stage}<br>"
                    f"Total sparsity: {val:.4f}<br>"
                    "<extra></extra>"
                ),
            ))

    N = df["N"].iloc[0]
    fig.update_layout(
        title={
            "text": f"Attention Sparsity After Bandwidth Propagation  (N={N})"
                    if show_title else None,
            "font": {"size": 18, "family": "Inter, Arial, sans-serif", "weight": "bold"},
            "x": 0.5, "xanchor": "center",
        },
        xaxis={
            "title": {"text": "<b>Propagation</b>",
                      "font": {"family": "Inter, Arial, sans-serif", "size": 13}},
            "tickmode": "array",
            "tickvals": list(range(n_masks)),
            "ticktext": mask_order,
            "tickfont": {"size": 11},
            "showgrid": False, "zeroline": False,
            "showline": True, "linewidth": 0.5, "linecolor": "black", "mirror": True,
        },
        yaxis={
            "title": {"text": "<b>Sparsity Ratio</b>",
                      "font": {"family": "Inter, Arial, sans-serif", "size": 13}},
            "range": [0, 1],
            "tickformat": ".2f",
            "tickfont": {"size": 11},
            "showgrid": True, "gridcolor": "lightgray", "gridwidth": 0.5,
            "zeroline": False,
            "showline": True, "linewidth": 0.5, "linecolor": "black", "mirror": True,
        },
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend={
            "orientation": "h", "yanchor": "top", "y": -0.18,
            "xanchor": "center", "x": 0.5,
            "bgcolor": "rgba(255,255,255,0.95)", "font": {"size": 11},
        },
        margin={"l": 0, "r": 20, "t": 50 if show_title else 20, "b": 0},
        width=700, height=420,
    )

    output_path = str(csv_path).split(".csv")[0]
    fig.write_html(
        f"{output_path}.html",
        config={"displayModeBar": True, "displaylogo": False},
    )
    if save_png_and_svg:
        try:
            fig.write_image(f"{output_path}.svg", width=550, height=410)
            fig.write_image(f"{output_path}.png", width=550, height=410, scale=2)
        except Exception as e:
            print(f"Image export skipped ({e}). Install kaleido: pip install kaleido")
    return fig


if __name__ == "__main__":
    create_plot()
