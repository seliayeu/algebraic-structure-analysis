import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import kaleido
import os


def bandwidth_plot(
    csv_path="./results/chain_matmul.csv",
    output_prefix="chain_matmul",
    figure_title: str = "Chain Matmul",
):
    """
    Create performance analysis plot for chain matrix multiplication.

    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing the data
    output_prefix : str
        Prefix for output files (will create {prefix}.html and {prefix}.svg)
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

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "<b>Execution Time vs Bands</b>",
            "<b>Memory Footprint vs Bands</b>",
        ),
        vertical_spacing=0.12,
        specs=[[{"type": "scatter"}], [{"type": "scatter"}]],
    )

    fig.add_trace(
        go.Scatter(
            x=baseline["bw"],
            y=baseline["avg_time_s"],
            mode="lines+markers",
            name="baseline",
            legendgroup="baseline",
            line=dict(color="#94A3B8", width=2.5, dash="solid"),
            marker=dict(
                size=9,
                symbol="circle",
                color="#94A3B8",
                line=dict(color="white", width=1.5),
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ar_dense["bw"],
            y=ar_dense["avg_time_s"],
            mode="lines+markers",
            name="bpa-dense",
            legendgroup="bpa-dense",
            line=dict(color="#F97316", width=3, shape="spline"),
            marker=dict(
                size=9,
                symbol="diamond",
                color="#F97316",
                line=dict(color="white", width=1.5),
            ),
            customdata=baseline.set_index("bw")
            .reindex(ar_dense["bw"])["avg_time_s"]
            .values
            / ar_dense["avg_time_s"].values,
            hovertemplate="<b>bpa-dense</b><br>Bands: %{x}<br>Time: %{y:.4f} seconds<br>Speedup: %{customdata:.1f}x<br><extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ar_dia["bw"],
            y=ar_dia["avg_time_s"],
            mode="lines+markers",
            name="bpa-dia",
            legendgroup="bpa-dia",
            line=dict(color="#06B6D4", width=3, shape="spline"),
            marker=dict(
                size=11,
                symbol="triangle-up",
                color="#06B6D4",
                line=dict(color="white", width=1.5),
            ),
            hovertemplate="<b>bpa-dia</b><br>Bands: %{x}<br>Time: %{y:.4f} seconds<br><extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ard_dia["bw"],
            y=ard_dia["avg_time_s"],
            mode="lines+markers",
            name="bpa-hybrid",
            legendgroup="bpa-hybrid",
            line=dict(color="#00b200", width=3, shape="spline"),
            marker=dict(
                size=9,
                symbol="square",
                color="#00b200",
                line=dict(color="white", width=1.5),
            ),
            hovertemplate="<b>bpa-hybrid</b><br>Bands: %{x}<br>Time: %{y:.4f} seconds<br><extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=baseline["bw"],
            y=baseline["max_rss_mb"],
            mode="lines+markers",
            name="baseline",
            legendgroup="baseline",
            showlegend=False,
            line=dict(color="#94A3B8", width=2.5),
            marker=dict(
                size=9,
                symbol="circle",
                color="#94A3B8",
                line=dict(color="white", width=1.5),
            ),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ar_dense["bw"],
            y=ar_dense["max_rss_mb"],
            mode="lines+markers",
            name="bpa-dense",
            legendgroup="bpa-dense",
            showlegend=False,
            line=dict(color="#F97316", width=3),
            marker=dict(
                size=9,
                symbol="diamond",
                color="#F97316",
                line=dict(color="white", width=1.5),
            ),
            # fill="tozeroy",
            # fillcolor="rgba(249, 115, 22, 0.1)",
            hovertemplate="<b>bpa-dense</b><br>Bands: %{x}<br>Memory: %{y:.1f} MB<br><extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ar_dia["bw"],
            y=ar_dia["max_rss_mb"],
            mode="lines+markers",
            name="bpa-dia",
            legendgroup="bpa-dia",
            showlegend=False,
            line=dict(color="#06B6D4", width=3),
            marker=dict(
                size=11,
                symbol="triangle-up",
                color="#06B6D4",
                line=dict(color="white", width=1.5),
            ),
            hovertemplate="<b>bpa-dia</b><br>Bands: %{x}<br>Memory: %{y:.1f} MB<br><extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=ard_dia["bw"],
            y=ard_dia["max_rss_mb"],
            mode="lines+markers",
            name="bpa-hybrid",
            legendgroup="bpa-hybrid",
            showlegend=False,
            line=dict(color="#00b200", width=3, shape="spline"),
            marker=dict(
                size=9,
                symbol="square",
                color="#00b200",
                line=dict(color="white", width=1.5),
            ),
            hovertemplate="<b>bpa-hybrid</b><br>Bands: %{x}<br>Memory: %{y:.1f} MB<br><extra></extra>",
        ),
        row=2,
        col=1,
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
            "orientation": "h",
            "yanchor": "top",
            "y": -0.12,
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
        width=900,
        height=900,
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
    )

    fig.update_yaxes(
        title_text="<b>Average Time (seconds)</b>",
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
        row=1,
        col=1,
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
        row=2,
        col=1,
    )

    image_output = str(csv_path).split(".csv")[0]
    fig.write_html(
        f"{image_output}.html",
        config={
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
            "displaylogo": False,
        },
    )
    fig.write_image(f"{image_output}.svg", width=1200, height=900)
    fig.write_image(f"{image_output}.png", width=1200, height=900, scale=2)

    # fig.show()

    print(f"Saved: {output_prefix}.html, {output_prefix}.svg, {output_prefix}.png")

    return fig

def compare_symbolic_chained(
    results_dir="./results",
    output_prefix="symbolic_chained_comparison",
    figure_title: str = "Symbolic Analysis Time vs Number of Matmuls",
    log = False,
):
    """
    Create performance analysis plot comparing DIA, Linalg, STUR, and TeSA
    symbolic execution times for chained matmuls.
    """
    
    dia_path = os.path.join(results_dir, "symbolic_chained_dia.csv")
    linalg_path = os.path.join(results_dir, "symbolic_chained_linalg.csv")
    stur_path = os.path.join(results_dir, "symbolic_chained_stur.csv")
    tesa_path = os.path.join(results_dir, "symbolic_chained_tesa.csv")

    try:
        df_dia = pd.read_csv(dia_path)
        df_linalg = pd.read_csv(linalg_path)
        df_stur = pd.read_csv(stur_path)
        df_tesa = pd.read_csv(tesa_path)
    except FileNotFoundError as e:
        print(f"Error loading CSVs: {e}")
        return

    df_dia["k"] = pd.to_numeric(df_dia["k"])
    df_linalg["k"] = pd.to_numeric(df_linalg["k"])
    df_stur["k"] = pd.to_numeric(df_stur["k"])
    df_tesa["k"] = pd.to_numeric(df_tesa["k"])

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_linalg["k"],
            y=df_linalg["avg_time_ms"],
            mode="lines+markers",
            name="BPA Linalg",
            legendgroup="linalg",
            line=dict(color="#94A3B8", width=2.5, dash="solid"),
            marker=dict(
                size=10,
                symbol="circle",
                color="#94A3B8",
                line=dict(color="white", width=1.5),
            ),
            hovertemplate="<b>BPA Linalg</b><br># Matmuls: %{x}<br>Time: %{y:.2f} ms<br><extra></extra>",
        )
    )

    speedup_dia = df_linalg.set_index("k").reindex(df_dia["k"])["avg_time_ms"].values / df_dia["avg_time_ms"].values

    fig.add_trace(
        go.Scatter(
            x=df_dia["k"],
            y=df_dia["avg_time_ms"],
            mode="lines+markers",
            name="BPA DIA",
            legendgroup="dia",
            line=dict(color="#F97316", width=3, shape="spline"),
            marker=dict(
                size=12,
                symbol="diamond",
                color="#F97316",
                line=dict(color="white", width=1.5),
            ),
            customdata=speedup_dia,
            hovertemplate="<b>BPA DIA</b><br># Matmuls: %{x}<br>Time: %{y:.2f} ms<br>Speedup (vs Linalg): %{customdata:.1f}x<br><extra></extra>",
        )
    )

    speedup_stur = df_linalg.set_index("k").reindex(df_stur["k"])["avg_time_ms"].values / df_stur["avg_time_ms"].values

    fig.add_trace(
        go.Scatter(
            x=df_stur["k"],
            y=df_stur["avg_time_ms"],
            mode="lines+markers",
            name="STUR",
            legendgroup="stur",
            line=dict(color="#06B6D4", width=3, shape="spline"),
            marker=dict(
                size=12,
                symbol="triangle-up",
                color="#06B6D4",
                line=dict(color="white", width=1.5),
            ),
            customdata=speedup_stur,
            hovertemplate="<b>STUR</b><br># Matmuls: %{x}<br>Time: %{y:.2f} ms<br>Speedup (vs Linalg): %{customdata:.1f}x<br><extra></extra>",
        )
    )

    speedup_tesa = df_linalg.set_index("k").reindex(df_tesa["k"])["avg_time_ms"].values / df_tesa["avg_time_ms"].values

    fig.add_trace(
        go.Scatter(
            x=df_tesa["k"],
            y=df_tesa["avg_time_ms"],
            mode="lines+markers",
            name="TeSA (1024x1024)",
            legendgroup="tesa",
            line=dict(color="#8B5CF6", width=3, shape="spline"),  # Using Purple to distinguish from others
            marker=dict(
                size=12,
                symbol="square",
                color="#8B5CF6",
                line=dict(color="white", width=1.5),
            ),
            customdata=speedup_tesa,
            hovertemplate="<b>TeSA</b><br># Matmuls: %{x}<br>Time: %{y:.2f} ms<br>Speedup (vs Linalg): %{customdata:.1f}x<br><extra></extra>",
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
        title_text="<b>Number of Matmuls</b>",
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
        type="log" if log else "linear",
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
    bandwidth_plot(csv_path="./results/chain_matmul.csv", output_prefix="chain_matmul", figure_title="Chain Matmul")
    bandwidth_plot(csv_path="./results/bertlike.csv", output_prefix="bertlike", figure_title="BERT-like")
    compare_symbolic_chained(log=True)
