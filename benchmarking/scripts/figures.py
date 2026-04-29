from typing import List
import pandas as pd
import plotly.graph_objects as go
import kaleido
import os


def create_comptime_plot(
    csv_path="./results/bpa_comptime.csv",
    output_prefix="bpa_comptime",
    figure_title: str = "Compilation Time Breakdown by Component",
    show_title: bool = True,
    save_png_and_svg: bool = True,
):

    df = pd.read_csv(csv_path)

    df_processed = df.copy()
    for idx, row in df_processed.iterrows():
        analysis_time = row["analysis_time_avg"]
        total_lowering = row["total_lowering_time_avg"]
        pure_lowering = total_lowering - analysis_time
        if pure_lowering < 0 and pure_lowering > -0.001:
            pure_lowering = 0
        df_processed.at[idx, "total_lowering_time"] = pure_lowering

    configs = df_processed["config"].unique()
    ops_values = [
        "kalman filter",
        "sparse attention",
        "batch bertlike",
        "chain100",
        "20",
        "30",
        "50",
        "100",
        "200",
    ]

    components = ["total_lowering_time", "analysis_time_avg", "runtime_avg"]
    component_labels = {
        "total_lowering_time": "Compilation Time",
        "analysis_time_avg": "Analysis Time",
        "runtime_avg": "Runtime",
    }

    COLOR_PALETTE = {
        "analysis_time_avg": "#e99575",
        "total_lowering_time": "#88CCEE",
        "runtime_avg": "#71b5a0",
    }

    fig = go.Figure()

    n_configs = len(configs)
    bar_width = 0.9 / n_configs

    for ops_idx, ops in enumerate(ops_values):
        group_center = ops_idx

        for config_idx, config in enumerate(configs):
            subset = df_processed[
                (df_processed["config"] == config) & (df_processed["ops"] == ops)
            ]

            if subset.empty:
                continue

            config_label = config.replace("_", " ").title()
            time_values = [subset[comp].values[0] for comp in components]
            x_position = group_center + (config_idx - n_configs / 2 + 0.5) * bar_width

            base_value = 0
            for comp_idx, (comp, time_val) in enumerate(zip(components, time_values)):
                fig.add_trace(
                    go.Bar(
                        name=component_labels[comp],
                        x=[x_position],
                        y=[time_val],
                        base=[base_value],
                        marker_color=COLOR_PALETTE[comp],
                        marker_line_color="black",
                        marker_line_width=0.8,
                        opacity=0.9,
                        width=bar_width * 0.85,
                        showlegend=(ops_idx == 0 and config_idx == 0),
                        legendgroup=comp,
                        hovertemplate=(
                            f"<b>{config_label}</b><br>"
                            + f"Benchmark: {ops}<br>"
                            + f"<b>{component_labels[comp]}</b>: {time_val:.4f} s<br>"
                            + f"<b>Total Time</b>: {sum(time_values):.4f} s<br>"
                            + "<extra></extra>"
                        ),
                    )
                )
                base_value += time_val

    x_tick_positions = list(range(len(ops_values)))
    x_tick_labels = ops_values

    layout_updates = {
        "title": {
            "text": figure_title if show_title else None,
            "font": {
                "size": 20,
                "family": "Inter, Arial, sans-serif",
                "weight": "bold",
            },
            "x": 0.5,
            "xanchor": "center",
        },
        "xaxis": {
            "title": {
                "text": "<b>Benchmark</b>",
                "font": {
                    "family": "Inter, Arial, sans-serif",
                    "size": 13,
                    "weight": "normal",
                },
            },
            "tickfont": {"size": 11},
            "tickmode": "array",
            "tickvals": x_tick_positions,
            "ticktext": x_tick_labels,
            "gridcolor": "lightgray",
            "gridwidth": 0.5,
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "linewidth": 0.5,
            "linecolor": "black",
            "mirror": True,
        },
        "yaxis": {
            "type": "log",
            "title": {
                "text": "<b>End-to-End Time (s) - Log Scale</b>",
                "font": {
                    "family": "Inter, Arial, sans-serif",
                    "size": 13,
                    "weight": "normal",
                },
            },
            "tickfont": {"size": 11},
            "gridcolor": "lightgray",
            "gridwidth": 0.5,
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "linewidth": 0.5,
            "linecolor": "black",
            "mirror": True,
        },
        "barmode": "stack",
        "plot_bgcolor": "white",
        "paper_bgcolor": "white",
        "legend": {
            "orientation": "h",
            "yanchor": "top",
            "y": -0.10,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255, 255, 255, 0.95)",
            "font": {"size": 11},
        },
        "margin": {"l": 0, "r": 20, "t": 0, "b": 0},
        "width": 700,
        "height": 500,
    }

    fig.update_layout(**layout_updates)

    output_path = str(csv_path).split(".csv")[0]
    fig.write_html(
        f"{output_path}.html",
        config={
            "displayModeBar": True,
            "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
            "displaylogo": False,
        },
    )
    if save_png_and_svg:
        fig.write_image(f"{output_path}.svg", width=550, height=410)
        fig.write_image(f"{output_path}.png", width=550, height=410, scale=2)

    fig.show()
    return fig


def create_time_plot(
    csv_path="./results/chain_matmul.csv",
    output_prefix="chain_matmul_time",
    figure_title: str = "Chain Matmul - Execution Time",
    show_title: bool = True,
    save_png_and_svg: bool = True,
    color_palette: List[str] = ["#94A3B8", "#8B5CF6", "#EC4899", "#3B82F6"],
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
            line=dict(color=color_palette[0], width=2, dash="solid"),
            marker=dict(
                size=6,
                symbol="circle",
                color=color_palette[0],
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
            line=dict(color=color_palette[1], width=2, shape="spline"),
            marker=dict(
                size=6,
                symbol="diamond",
                color=color_palette[1],
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
            line=dict(color=color_palette[2], width=2, shape="spline"),
            marker=dict(
                size=9,
                symbol="triangle-up",
                color=color_palette[2],
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
            line=dict(color=color_palette[3], width=2, shape="spline"),
            marker=dict(
                size=6,
                symbol="square",
                color=color_palette[3],
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
            "y": -0.13,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255, 255, 255, 0.95)",
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
        margin=dict(t=0, b=0, l=0, r=20),
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        title_text="<b>Bandwidth</b>",
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
    if save_png_and_svg:
        fig.write_image(f"{output_path}.svg", width=550, height=410)
        fig.write_image(f"{output_path}.png", width=550, height=410, scale=2)

    fig.show()
    print(f"Saved time plot: {output_path}.html, .svg, .png")
    return fig


def create_memory_plot(
    csv_path="./results/chain_matmul.csv",
    output_prefix="chain_matmul_memory",
    figure_title: str = "Chain Matmul - Memory Footprint",
    show_title: bool = True,
    save_png_and_svg: bool = True,
    color_palette: List[str] = ["#94A3B8", "#8B5CF6", "#EC4899", "#3B82F6"],
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
            line=dict(color=color_palette[0], width=2, dash="solid"),
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
            line=dict(color=color_palette[1], width=2, shape="spline"),
            marker=dict(
                size=6,
                symbol="diamond",
                color=color_palette[1],
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
            line=dict(color=color_palette[2], width=2, shape="spline"),
            marker=dict(
                size=9,
                symbol="triangle-up",
                color=color_palette[2],
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
            line=dict(color=color_palette[3], width=2, shape="spline"),
            marker=dict(
                size=6,
                symbol="square",
                color=color_palette[3],
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
            "y": -0.13,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255, 255, 255, 0.95)",
            "font": {"size": 11},
        },
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Inter, monospace",
        ),
        width=700,
        height=500,
        template="plotly_white",
        margin=dict(t=0, b=0, l=0, r=20),
        plot_bgcolor="white",
    )

    fig.update_xaxes(
        title_text="<b>Bandwidth</b>",
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
        showline=True,
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
    if save_png_and_svg:
        fig.write_image(f"{output_path}.svg", width=550, height=430)
        fig.write_image(f"{output_path}.png", width=550, height=430, scale=2)

    fig.show()
    print(f"Saved memory plot: {output_path}.html, .svg, .png")
    return fig


def bandwidth_plot(
    csv_path="./results/chain_matmul.csv",
    output_prefix="chain_matmul",
    figure_title: str = "Chain Matmul",
    show_title: bool = True,
    save_png_and_svg: bool = True,
    color_palette: List[str] = ["#94A3B8", "#8B5CF6", "#EC4899", "#3B82F6"],
):
    """
    Create performance analysis plots for chain matrix multiplication.
    Creates two separate figures: one for execution time and one for memory footprint.
    """

    time_fig = create_time_plot(
        csv_path,
        output_prefix,
        f"{figure_title} - Execution Time",
        show_title,
        save_png_and_svg,
        color_palette,
    )
    memory_fig = create_memory_plot(
        csv_path,
        output_prefix,
        f"{figure_title} - Memory Footprint",
        show_title,
        save_png_and_svg,
        color_palette,
    )

    return time_fig, memory_fig


def compare_symbolic_chained(
    results_dir="./results",
    output_prefix="symbolic_chained_comparison",
    figure_title: str = "Symbolic Analysis Time vs Number of Matmuls",
    log=False,
    show_title: bool = True,
    save_png_and_svg: bool = True,
):
    """
    Create performance analysis plot comparing averaged BPA, STUR, and SparTA
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

    df_bpa = pd.merge(
        df_dia[["k", "avg_time_ms"]],
        df_linalg[["k", "avg_time_ms"]],
        on="k",
        suffixes=("_dia", "_linalg"),
    )
    df_bpa["avg_time_ms"] = (
        df_bpa["avg_time_ms_dia"] + df_bpa["avg_time_ms_linalg"]
    ) / 2

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_bpa["k"],
            y=df_bpa["avg_time_ms"],
            mode="lines+markers",
            name="bpa",
            legendgroup="bpa",
            line=dict(color="#06B6D4", width=2, shape="spline"),
            marker=dict(
                size=9,
                symbol="triangle-up",
                color="#06B6D4",
                line=dict(color="white", width=1.5),
            ),
            hovertemplate="<b>bpa</b><br># Matmuls: %{x}<br>Time: %{y:.2f} ms<br><extra></extra>",
        )
    )

    speedup_stur = (
        df_bpa.set_index("k").reindex(df_stur["k"])["avg_time_ms"].values
        / df_stur["avg_time_ms"].values
    )

    fig.add_trace(
        go.Scatter(
            x=df_stur["k"],
            y=df_stur["avg_time_ms"],
            mode="lines+markers",
            name="stur",
            legendgroup="stur",
            line=dict(color="#8d1261", width=2, shape="spline"),
            marker=dict(
                size=6,
                symbol="circle",
                color="#8d1261",
                line=dict(color="white", width=1.5),
            ),
            customdata=speedup_stur,
            hovertemplate="<b>stur</b><br># Matmuls: %{x}<br>Time: %{y:.2f} ms<br>Speedup (vs bpa): %{customdata:.1f}x<br><extra></extra>",
        )
    )

    speedup_tesa = (
        df_bpa.set_index("k").reindex(df_tesa["k"])["avg_time_ms"].values
        / df_tesa["avg_time_ms"].values
    )

    fig.add_trace(
        go.Scatter(
            x=df_tesa["k"],
            y=df_tesa["avg_time_ms"],
            mode="lines+markers",
            name="sparta",
            legendgroup="sparta",
            line=dict(color="#8B5CF6", width=2, shape="spline"),
            marker=dict(
                size=6,
                symbol="square",
                color="#8B5CF6",
                line=dict(color="white", width=1.5),
            ),
            customdata=speedup_tesa,
            hovertemplate="<b>sparta</b><br># Matmuls: %{x}<br>Time: %{y:.2f} ms<br>Speedup (vs bpa): %{customdata:.1f}x<br><extra></extra>",
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
            "y": -0.13,
            "xanchor": "center",
            "x": 0.5,
            "bgcolor": "rgba(255, 255, 255, 0.95)",
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
        margin=dict(t=0, b=0, l=0, r=0),
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
        type="log" if log else "linear",
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
    if save_png_and_svg:
        fig.write_image(f"{output_path}.svg", width=550, height=410)
        fig.write_image(f"{output_path}.png", width=550, height=410, scale=2)

    fig.show()
    print(
        f"Saved analysis plot: {output_path}.html"
        + (", .svg, .png" if save_png_and_svg else "")
    )

    return fig


if __name__ == "__main__":
    create_memory_plot(
        csv_path="./results/sparse_attention.csv",
        output_prefix="kalman_filter",
        figure_title="kalman filter",
        show_title=False,
        save_png_and_svg=False,
    )
    # bandwidth_plot(
    #     csv_path="./results/batch_bertlike.csv",
    #     output_prefix="batch_bertlike",
    #     figure_title="BERT-like",
    #     show_title=False,
    # )
    # bandwidth_plot(csv_path="./results/bertlike.csv", output_prefix="bertlike", figure_title="BERT-like")
    # compare_symbolic_chained(log=True, show_title=False)
