import pandas as pd
import plotly.graph_objects as go
import kaleido
import os


def create_time_plot(
    csv_path="./results/chain_matmul.csv",
    output_prefix="chain_matmul_time",
    figure_title: str = "Chain Matmul - Execution Time",
    show_title: bool = True,
):
    """
    Create execution time plot for chain matrix multiplication.
    """
    df = pd.read_csv(csv_path)
    df["bw"] = pd.to_numeric(df["bw"])

    max_bw = df["bw"].max()
    x_max = max_bw + 10

    baseline = df[df["config"] == "baseline"]
    ar_dense = df[
        (df["config"] == "AR") & (df["file_name"].str.contains("dense", na=False))
    ]
    ar_dia = df[
        (df["config"] == "AR") & (df["file_name"].str.contains("dia", na=False))
    ]
    ard_dia = df[
        (df["config"] == "ADR") & (df["file_name"].str.contains("dia", na=False))
    ]
    x_range = [-10, x_max]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=baseline["bw"],
            y=baseline["avg_time_s"],
            mode="lines+markers",
            name="baseline",
            legendgroup="baseline",
            line=dict(color="#94A3B8", width=2, dash="solid"),
            marker=dict(
                size=6,
                symbol="circle",
                color="#94A3B8",
                line=dict(color="white", width=1.5),
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ar_dense["bw"],
            y=ar_dense["avg_time_s"],
            mode="lines+markers",
            name="bpa-dense",
            legendgroup="bpa-dense",
            line=dict(color="#F97316", width=2, shape="spline"),
            marker=dict(
                size=6,
                symbol="diamond",
                color="#F97316",
                line=dict(color="white", width=1.5),
            ),
            customdata=baseline.set_index("bw")
            .reindex(ar_dense["bw"])["avg_time_s"]
            .values
            / ar_dense["avg_time_s"].values,
            hovertemplate="<b>bpa-dense</b><br>Bands: %{x}<br>Time: %{y:.4f} seconds<br>Speedup: %{customdata:.1f}x<br><extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ar_dia["bw"],
            y=ar_dia["avg_time_s"],
            mode="lines+markers",
            name="bpa-dia",
            legendgroup="bpa-dia",
            line=dict(color="#06B6D4", width=2, shape="spline"),
            marker=dict(
                size=9,
                symbol="triangle-up",
                color="#06B6D4",
                line=dict(color="white", width=1.5),
            ),
            hovertemplate="<b>bpa-dia</b><br>Bands: %{x}<br>Time: %{y:.4f} seconds<br><extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ard_dia["bw"],
            y=ard_dia["avg_time_s"],
            mode="lines+markers",
            name="bpa-hybrid",
            legendgroup="bpa-hybrid",
            line=dict(color="#00b200", width=2, shape="spline"),
            marker=dict(
                size=6,
                symbol="square",
                color="#00b200",
                line=dict(color="white", width=1.5),
            ),
            hovertemplate="<b>bpa-hybrid</b><br>Bands: %{x}<br>Time: %{y:.4f} seconds<br><extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": figure_title if show_title else "",
            "font": {
                "size": 20,
                "family": "Inter, Arial, sans-serif",
                "weight": "bold",
            },
            "x": 0.5,
            "xanchor": "center",
        },
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.17,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255, 255, 255, 0.95)",
            "bordercolor": "#E2E8F0",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, monospace",
            bordercolor="#CBD5E1",
        ),
        width=700,
        height=500,
        template="plotly_white",
        margin=dict(t=100, l=80, r=60, b=120),
        paper_bgcolor="#F8FAFC",
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        title_text="<b>Bands (bw)</b>",
        title_font=dict(size=13, family="Inter, Arial, sans-serif"),
        tickfont=dict(size=11),
        gridcolor="#E2E8F0",
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor="#CBD5E1",
        linewidth=1,
        mirror=True,
        range=x_range,
    )

    fig.update_yaxes(
        type="log",
        title_text="<b>Avg. Time (s) - Log Scale</b>",
        title_font=dict(size=13, family="Inter, Arial, sans-serif"),
        tickfont=dict(size=11),
        gridcolor="#E2E8F0",
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor="#CBD5E1",
        linewidth=1,
        mirror=True,
    )

    output_path = str(csv_path).split(".csv")[0] + "_time"
    fig.write_html(
        f"{output_path}.html",
        config={
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
            "displaylogo": False,
        },
    )
    fig.write_image(f"{output_path}.svg", width=700, height=500)
    fig.write_image(f"{output_path}.png", width=700, height=500, scale=2)

    fig.show()
    print(f"Saved time plot: {output_path}.html, .svg, .png")
    return fig


def create_memory_plot(
    csv_path="./results/chain_matmul.csv",
    output_prefix="chain_matmul_memory",
    figure_title: str = "Chain Matmul - Memory Footprint",
    show_title: bool = True,
):
    """
    Create memory footprint plot for chain matrix multiplication.
    """
    df = pd.read_csv(csv_path)
    df["bw"] = pd.to_numeric(df["bw"])

    baseline = df[df["config"] == "baseline"]
    ar_dense = df[
        (df["config"] == "AR") & (df["file_name"].str.contains("dense", na=False))
    ]
    ar_dia = df[
        (df["config"] == "AR") & (df["file_name"].str.contains("dia", na=False))
    ]
    ard_dia = df[
        (df["config"] == "ADR") & (df["file_name"].str.contains("dia", na=False))
    ]

    max_bw = df["bw"].max()
    x_max = max_bw + 10

    x_range = [-10, x_max]
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=baseline["bw"],
            y=baseline["max_rss_mb"],
            mode="lines+markers",
            name="baseline",
            legendgroup="baseline",
            line=dict(color="#94A3B8", width=2, dash="solid"),
            marker=dict(
                size=6,
                symbol="circle",
                color="#94A3B8",
                line=dict(color="white", width=1.5),
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ar_dense["bw"],
            y=ar_dense["max_rss_mb"],
            mode="lines+markers",
            name="bpa-dense",
            legendgroup="bpa-dense",
            line=dict(color="#F97316", width=2, shape="spline"),
            marker=dict(
                size=6,
                symbol="diamond",
                color="#F97316",
                line=dict(color="white", width=1.5),
            ),
            hovertemplate="<b>bpa-dense</b><br>Bands: %{x}<br>Memory: %{y:.1f} MB<br><extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ar_dia["bw"],
            y=ar_dia["max_rss_mb"],
            mode="lines+markers",
            name="bpa-dia",
            legendgroup="bpa-dia",
            line=dict(color="#06B6D4", width=2, shape="spline"),
            marker=dict(
                size=9,
                symbol="triangle-up",
                color="#06B6D4",
                line=dict(color="white", width=1.5),
            ),
            hovertemplate="<b>bpa-dia</b><br>Bands: %{x}<br>Memory: %{y:.1f} MB<br><extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=ard_dia["bw"],
            y=ard_dia["max_rss_mb"],
            mode="lines+markers",
            name="bpa-hybrid",
            legendgroup="bpa-hybrid",
            line=dict(color="#00b200", width=2, shape="spline"),
            marker=dict(
                size=6,
                symbol="square",
                color="#00b200",
                line=dict(color="white", width=1.5),
            ),
            hovertemplate="<b>bpa-hybrid</b><br>Bands: %{x}<br>Memory: %{y:.1f} MB<br><extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": figure_title if show_title else "",
            "font": {
                "size": 20,
                "family": "Inter, Arial, sans-serif",
                "weight": "bold",
            },
            "x": 0.5,
            "xanchor": "center",
        },
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.17,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255, 255, 255, 0.95)",
            "bordercolor": "#E2E8F0",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, monospace",
            bordercolor="#CBD5E1",
        ),
        width=700,
        height=500,
        template="plotly_white",
        margin=dict(t=100, l=80, r=60, b=120),
        paper_bgcolor="#F8FAFC",
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        title_text="<b>Bands (bw)</b>",
        title_font=dict(size=13, family="Inter, Arial, sans-serif"),
        tickfont=dict(size=11),
        gridcolor="#E2E8F0",
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor="#CBD5E1",
        linewidth=1,
        mirror=True,
        range=x_range,
    )

    fig.update_yaxes(
        title_text="<b>Max RSS (MB)</b>",
        title_font=dict(size=13, family="Inter, Arial, sans-serif"),
        tickfont=dict(size=11),
        gridcolor="#E2E8F0",
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor="#CBD5E1",
        linewidth=1,
        mirror=True,
    )

    output_path = str(csv_path).split(".csv")[0] + "_memory"
    fig.write_html(
        f"{output_path}.html",
        config={
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
            "displaylogo": False,
        },
    )
    fig.write_image(f"{output_path}.svg", width=700, height=500)
    fig.write_image(f"{output_path}.png", width=700, height=500, scale=2)

    fig.show()
    print(f"Saved memory plot: {output_path}.html, .svg, .png")
    return fig


def bandwidth_plot(
    csv_path="./results/chain_matmul.csv",
    output_prefix="chain_matmul",
    figure_title: str = "Chain Matmul",
    show_title: bool = True,
):
    """
    Create performance analysis plots for chain matrix multiplication.
    Creates two separate figures: one for execution time and one for memory footprint.
    """
    # Create both plots
    time_fig = create_time_plot(
        csv_path, output_prefix, f"{figure_title} - Execution Time", show_title
    )
    memory_fig = create_memory_plot(
        csv_path, output_prefix, f"{figure_title} - Memory Footprint", show_title
    )

    return time_fig, memory_fig


def compare_symbolic_chained(
    results_dir="./results",
    output_prefix="symbolic_chained_comparison",
    figure_title: str = "Symbolic Analysis Time vs Number of Matmuls",
):
    """
    Create performance analysis plot comparing DIA, Linalg, and STUR
    symbolic execution times for chained matmuls.
    """

    dia_path = os.path.join(results_dir, "symbolic_chained_dia.csv")
    linalg_path = os.path.join(results_dir, "symbolic_chained_linalg.csv")
    stur_path = os.path.join(results_dir, "symbolic_chained_stur.csv")

    try:
        df_dia = pd.read_csv(dia_path)
        df_linalg = pd.read_csv(linalg_path)
        df_stur = pd.read_csv(stur_path)
    except FileNotFoundError as e:
        print(f"Error loading CSVs: {e}")
        return

    df_dia["k"] = pd.to_numeric(df_dia["k"])
    df_linalg["k"] = pd.to_numeric(df_linalg["k"])
    df_stur["k"] = pd.to_numeric(df_stur["k"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_linalg["k"],
            y=df_linalg["avg_time_ms"],
            mode="lines+markers",
            name="MLIR Linalg (Dense)",
            legendgroup="linalg",
            line=dict(color="#94A3B8", width=2.5, dash="solid"),
            marker=dict(
                size=10,
                symbol="circle",
                color="#94A3B8",
                line=dict(color="white", width=1.5),
            ),
            hovertemplate="<b>MLIR Linalg</b><br>K (Number of Matmuls): %{x}<br>Time: %{y:.2f} ms<br><extra></extra>",
        )
    )

    speedup_dia = (
        df_linalg.set_index("k").reindex(df_dia["k"])["avg_time_ms"].values
        / df_dia["avg_time_ms"].values
    )

    fig.add_trace(
        go.Scatter(
            x=df_dia["k"],
            y=df_dia["avg_time_ms"],
            mode="lines+markers",
            name="MLIR DIA (Banded)",
            legendgroup="dia",
            line=dict(color="#F97316", width=3, shape="spline"),
            marker=dict(
                size=12,
                symbol="diamond",
                color="#F97316",
                line=dict(color="white", width=1.5),
            ),
            customdata=speedup_dia,
            hovertemplate="<b>MLIR DIA</b><br>K (# Matmuls): %{x}<br>Time: %{y:.2f} ms<br>Speedup (vs Linalg): %{customdata:.1f}x<br><extra></extra>",
        )
    )

    speedup_stur = (
        df_linalg.set_index("k").reindex(df_stur["k"])["avg_time_ms"].values
        / df_stur["avg_time_ms"].values
    )

    fig.add_trace(
        go.Scatter(
            x=df_stur["k"],
            y=df_stur["avg_time_ms"],
            mode="lines+markers",
            name="STUR (Scala)",
            legendgroup="stur",
            line=dict(color="#06B6D4", width=3, shape="spline"),
            marker=dict(
                size=12,
                symbol="triangle-up",
                color="#06B6D4",
                line=dict(color="white", width=1.5),
            ),
            customdata=speedup_stur,
            hovertemplate="<b>STUR</b><br>K (# Matmuls): %{x}<br>Time: %{y:.2f} ms<br>Speedup (vs Linalg): %{customdata:.1f}x<br><extra></extra>",
        )
    )

    fig.update_layout(
        title={
            "text": figure_title,
            "font": {
                "size": 20,
                "family": "Inter, Arial, sans-serif",
                "weight": "bold",
            },
            "x": 0.5,
            "xanchor": "center",
        },
        showlegend=True,
        legend={
            "title": {"text": "Configuration", "font": {"size": 12}},
            "orientation": "h",
            "yanchor": "top",
            "y": -0.15,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255, 255, 255, 0.95)",
            "bordercolor": "#E2E8F0",
            "borderwidth": 1,
            "font": {"size": 11},
        },
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, monospace",
            bordercolor="#CBD5E1",
        ),
        width=1200,
        height=700,
        template="plotly_white",
        margin=dict(t=100, l=80, r=60, b=120),
        paper_bgcolor="#F8FAFC",
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        title_text="<b>Number of Matmuls (K)</b>",
        title_font=dict(size=13, family="Inter, Arial, sans-serif"),
        tickfont=dict(size=11),
        gridcolor="#E2E8F0",
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor="#CBD5E1",
        linewidth=1,
        mirror=True,
    )

    fig.update_yaxes(
        title_text="<b>Average Time (ms)</b>",
        title_font=dict(size=13, family="Inter, Arial, sans-serif"),
        tickfont=dict(size=11),
        gridcolor="#E2E8F0",
        gridwidth=1,
        showgrid=True,
        zeroline=False,
        showline=True,
        linecolor="#CBD5E1",
        linewidth=1,
        mirror=True,
    )

    output_path = os.path.join(results_dir, output_prefix)

    fig.write_html(
        f"{output_path}.html",
        config={
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
            "displaylogo": False,
        },
    )
    fig.write_image(f"{output_path}.svg", width=1200, height=700)
    fig.write_image(f"{output_path}.png", width=1200, height=700, scale=2)

    print(f"Saved: {output_path}.html, .svg, .png")

    return fig


if __name__ == "__main__":
    bandwidth_plot(
        csv_path="./results/chain_matmul.csv",
        output_prefix="chain_matmul",
        figure_title="Chain Matmul",
        show_title=False,
    )
    # bandwidth_plot(csv_path="./results/bertlike.csv", output_prefix="bertlike", figure_title="BERT-like")
    # compare_symbolic_chained()
