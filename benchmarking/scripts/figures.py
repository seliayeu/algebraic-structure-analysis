import kaleido
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional

pattern_map = {
    "dense": "x",
    "dia": "/",
    "hybrid": "\\",
    "dia-inputs": "+",
    "hybrid-dia-inputs": "|",
}


def create_comptime_plot(
    csv_path="./results/compilation_time.csv",
    output_prefix="comp_time",
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

    pattern_map = {
        "total_lowering_time": "x",
        "analysis_time_avg": "/",
        "runtime_avg": "\\",
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
                        marker_pattern_shape=pattern_map[comp],
                        marker_pattern_solidity=0.1,
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
    ops_names = [
        "Kalman<br>Filter",
        "Sparse<br>Attention",
        "Bertlike",
        "Chain<br>Matmul",
        "20",
        "30",
        "50",
        "100",
        "200",
    ]
    x_tick_labels = ops_names

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
            "y": -0.16,
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


def create_grouped_memory_plot(
    csv_paths: List[str],
    output_prefix: str = "memory_consumption",
    figure_title: str = "Bandwidth Memory Consumption",
    show_title: bool = True,
    save_png_and_svg: bool = True,
    color_palette: Optional[Dict[str, str]] = None,
):
    color_palette = {
        "dia-inputs": "rgb(229,196,148)",
        "hybrid-dia-inputs": "rgb(217,95,2)",
        # "baseline": "#ef4444",
        "baseline": "#7f7f7f",
        "dense": "#A5B4FC",
        "dia": "#C4B5FD",
        "hybrid": "rgb(77,175,74)",
    }

    n_rows = 2
    n_cols = 2

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_yaxes=False,
        shared_xaxes=False,
        horizontal_spacing=0.06,
        vertical_spacing=0.01,
    )

    bar_width = 0.25

    for idx, csv_path in enumerate(csv_paths):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        df = pd.read_csv(csv_path)
        df["bw"] = pd.to_numeric(df["bw"])

        df["max_rss_gb"] = df["max_rss_mb"] / 1024

        benchmark_name = (
            csv_path.split("/")[-1].replace(".csv", "").replace("_", " ").title()
        )

        baseline = df[df["config"] == "baseline"]

        ar_dense = df[
            (df["config"] == "AR") & (df["file_name"].str.contains("dense", na=False))
        ]
        ar_dia = df[
            (df["config"] == "AR")
            & (df["file_name"].str.contains("dia.mlir", na=False))
        ]
        ard_dia = df[
            (df["config"] == "ADR")
            & (df["file_name"].str.contains("dia.mlir", na=False))
        ]
        ar_dia_inputs = df[
            (df["config"] == "AR")
            & (df["file_name"].str.contains("dia_inputs.mlir", na=False))
        ]
        ard_dia_inputs = df[
            (df["config"] == "ADR")
            & (df["file_name"].str.contains("dia_inputs.mlir", na=False))
        ]

        group_space = 1.5
        bw_values = sorted(df["bw"].unique())
        n_bandwidths = len(bw_values)

        if not baseline.empty:
            baseline_mean = baseline["max_rss_gb"].mean()

            x_min = -0.5
            x_max = group_space * n_bandwidths - 0.65

            fig.add_trace(
                go.Scatter(
                    x=[x_min, x_max],
                    y=[baseline_mean, baseline_mean],
                    mode="lines",
                    name="baseline",
                    legend="legend1",
                    legendgroup="baseline",
                    line=dict(color=color_palette["baseline"], width=2.5, dash="dash"),
                    hovertemplate=f"<b>baseline</b><br>Mean: {baseline_mean:.1f} MB<br><extra></extra>",
                    showlegend=(idx == 0),
                ),
                row=row,
                col=col,
            )

        bar_configs = [
            ("dense", ar_dense, color_palette["dense"]),
            ("dia", ar_dia, color_palette["dia"]),
            ("hybrid", ard_dia, color_palette["hybrid"]),
            ("dia-inputs", ar_dia_inputs, color_palette["dia-inputs"]),
            (
                "hybrid-dia-inputs",
                ard_dia_inputs,
                color_palette["hybrid-dia-inputs"],
            ),
        ]

        for config_idx, (name, data, color) in enumerate(bar_configs):
            if data.empty:
                continue

            bw_to_memory = dict(zip(data["bw"], data["max_rss_gb"]))

            x_pos = []
            y_values = []
            customdata = []

            for bw_idx, bw in enumerate(bw_values):
                if bw in bw_to_memory:
                    x_pos.append(bw_idx * group_space + (config_idx - 1) * bar_width)
                    y_values.append(bw_to_memory[bw])
                    customdata.append(bw)

            if x_pos:
                fig.add_trace(
                    go.Bar(
                        x=x_pos,
                        y=y_values,
                        name=name,
                        legendgroup=name,
                        marker_color=color,
                        marker_line_color="black",
                        marker_line_width=0.8,
                        opacity=0.85,
                        width=bar_width,
                        hovertemplate=f"<b>{name}</b><br>Bandwidth: %{{customdata}}<br>Memory: %{{y:.1f}} MB<br><extra></extra>",
                        customdata=customdata,
                        legend="legend2",
                        showlegend=(idx == 0),
                        marker_pattern_shape=pattern_map[name],
                        marker_pattern_solidity=0.1,
                    ),
                    row=row,
                    col=col,
                )

        tick_positions = [i * group_space for i in range(n_bandwidths)]
        tick_labels = [str(bw) for bw in bw_values]

        fig.update_xaxes(
            # title_text="Bandwidth" if row == 2 else "",
            title_font={"family": "serif", "size": 12, "weight": "normal"},
            tickfont={"family": "serif", "size": 12},
            tickmode="array",
            tickvals=tick_positions if row == 2 else [],
            ticktext=tick_labels if row == 2 else None,
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=0.5,
            tickangle=45,
            linecolor="black",
            mirror=True,
            row=row,
            col=col,
            title_standoff=10,
            # side="top",
        )

        fig.update_yaxes(
            # title_text="Peak Memory Consumption (MB)" if col == 1 else "",
            title_font={"family": "serif", "size": 10, "weight": "normal"},
            tickfont={"family": "serif", "size": 12},
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=0.5,
            linecolor="black",
            mirror=True,
            row=row,
            col=col,
        )

        max_y = df["max_rss_gb"].max()
        fig.add_annotation(
            text=" " + benchmark_name + " ",
            x=2.0,
            y=max_y * 0.9999,
            xref=f"x{idx + 1}",
            yref=f"y{idx + 1}",
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font={"family": "serif", "size": 12, "weight": "bold", "color": "#333333"},
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=0.5,
            row=row,
            col=col,
        )

    fig.add_annotation(
        text="<b>Bandwidth</b>",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.11,
        xanchor="center",
        yanchor="middle",
        showarrow=False,
        font={"family": "serif", "size": 12, "weight": "normal"},
    )

    fig.add_annotation(
        text="<b>Peak Memory Consumption (GB)</b>",
        xref="paper",
        yref="paper",
        x=-0.08,
        y=0.5,
        textangle=-90,
        xanchor="center",
        yanchor="middle",
        showarrow=False,
        font={"family": "serif", "size": 12, "weight": "normal"},
    )

    fig.update_layout(
        title={
            "text": figure_title if show_title else None,
            "font": {"family": "serif", "size": 12, "weight": "normal"},
            "x": 0.5,
            "xanchor": "center",
        },
        showlegend=True,
        legend1={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.22,
            "xanchor": "center",
            "x": 0.5,
            "font": {"family": "serif", "size": 12},
            "bgcolor": "rgba(255, 255, 255, 0.95)",
        },
        legend2={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.30,
            "xanchor": "center",
            "x": 0.5,
            "font": {"family": "serif", "size": 12},
            "bgcolor": "rgba(255, 255, 255, 0.95)",
        },
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="serif",
        ),
        width=900,
        height=700,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=0, b=0, l=40, r=20),
    )

    output_path = "./results/memory_comparison"
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


def create_grouped_runtime_plot(
    csv_paths: List[str],
    output_prefix: str = "timing_comparison",
    figure_title: str = "Average Runtime Comparison",
    show_title: bool = True,
    save_png_and_svg: bool = True,
    color_palette: Optional[Dict[str, str]] = None,
):
    color_palette = {
        "baseline": "#7f7f7f",
        "baseline_opt": "#ef4444",
        "dense": "#F4A460",
        "dia": "#87CEEB",
        "hybrid": "rgb(77,175,74)",
        "dia-inputs": "#BDA6CE",
        "hybrid-dia-inputs": "#6d29ff",
    }

    n_rows = 2
    n_cols = 2

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        shared_yaxes=False,
        shared_xaxes=False,
        horizontal_spacing=0.05,
        vertical_spacing=0.01,
    )

    bar_width = 0.25

    for idx, csv_path in enumerate(csv_paths):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        df = pd.read_csv(csv_path)
        df["bw"] = pd.to_numeric(df["bw"])

        benchmark_name = (
            csv_path.split("/")[-1].replace(".csv", "").replace("_", " ").title()
        )

        baseline = df[df["config"] == "baseline"]
        baseline_opt = df[(df["config"] == "baseline_opt")]

        ar_dense = df[
            (df["config"] == "AR") & (df["file_name"].str.contains("dense", na=False))
        ]
        ar_dia = df[
            (df["config"] == "AR")
            & (df["file_name"].str.contains("dia.mlir", na=False))
        ]
        ard_dia = df[
            (df["config"] == "ADR")
            & (df["file_name"].str.contains("dia.mlir", na=False))
        ]
        ar_dia_inputs = df[
            (df["config"] == "AR")
            & (df["file_name"].str.contains("dia_inputs.mlir", na=False))
        ]
        ard_dia_inputs = df[
            (df["config"] == "ADR")
            & (df["file_name"].str.contains("dia_inputs.mlir", na=False))
        ]

        bw_values = sorted(df["bw"].unique())
        n_bandwidths = len(bw_values)

        group_space = 1.5
        x_min = -0.5
        x_max = group_space * n_bandwidths - 0.65

        # baseline
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[baseline["avg_time_s"].values[0], baseline["avg_time_s"].values[0]],
                mode="lines",
                name="baseline",
                legendgroup="baseline",
                legend="legend1",
                line=dict(color=color_palette["baseline"], width=2.0, dash="dash"),
                hovertemplate=f"<b>baseline</b><br>Time: {baseline['avg_time_s'].values[0]:.4f} s<br><extra></extra>",
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )
        # baseline opt
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[
                    baseline_opt["avg_time_s"].values[0],
                    baseline_opt["avg_time_s"].values[0],
                ],
                mode="lines",
                name="baseline-ikj",
                legend="legend1",
                legendgroup="baseline_opt",
                line=dict(color=color_palette["baseline_opt"], width=2.0, dash="dot"),
                hovertemplate=f"<b>baseline_opt</b><br>Time: {baseline_opt['avg_time_s'].values[0]:.4f} s<br><extra></extra>",
                showlegend=(idx == 0),
            ),
            row=row,
            col=col,
        )

        bar_configs = [
            ("dense", ar_dense, color_palette["dense"]),
            ("dia", ar_dia, color_palette["dia"]),
            ("hybrid", ard_dia, color_palette["hybrid"]),
            ("dia-inputs", ar_dia_inputs, color_palette["dia-inputs"]),
            (
                "hybrid-dia-inputs",
                ard_dia_inputs,
                color_palette["hybrid-dia-inputs"],
            ),
        ]

        for config_idx, (name, data, color) in enumerate(bar_configs):
            if data.empty:
                continue

            bw_to_time = dict(zip(data["bw"], data["avg_time_s"]))

            x_pos = []
            y_values = []
            customdata = []

            for bw_idx, bw in enumerate(bw_values):
                if bw in bw_to_time:
                    x_pos.append(bw_idx * group_space + (config_idx - 1) * bar_width)
                    y_values.append(bw_to_time[bw])
                    customdata.append(bw)

            if x_pos:
                fig.add_trace(
                    go.Bar(
                        x=x_pos,
                        y=y_values,
                        name=name,
                        legendgroup=name,
                        marker_color=color,
                        marker_line_color="black",
                        marker_line_width=0.8,
                        opacity=0.85,
                        width=bar_width,
                        hovertemplate=f"<b>{name}</b><br>Bandwidth: %{{customdata}}<br>Time: %{{y:.4f}} s<br><extra></extra>",
                        customdata=customdata,
                        legend="legend2",
                        showlegend=(idx == 0),
                        marker_pattern_shape=pattern_map[name],
                        marker_pattern_solidity=0.1,
                    ),
                    row=row,
                    col=col,
                )

        tick_positions = [i * group_space for i in range(n_bandwidths)]
        tick_labels = [str(bw) for bw in bw_values]

        fig.update_xaxes(
            title_font={"family": "serif", "size": 12, "weight": "normal"},
            tickfont={"family": "serif", "size": 12},
            tickmode="array",
            tickvals=tick_positions if row == 2 else [],
            ticktext=tick_labels if row == 2 else None,
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=0.5,
            linecolor="black",
            mirror=True,
            row=row,
            tickangle=45,
            col=col,
            title_standoff=10,
        )

        fig.update_yaxes(
            type="log",
            title_font={"family": "serif", "size": 10, "weight": "normal"},
            tickfont={"family": "serif", "size": 12},
            showgrid=False,
            zeroline=False,
            showline=True,
            linewidth=0.5,
            linecolor="black",
            mirror=True,
            row=row,
            col=col,
        )

        fig.add_annotation(
            text=" " + benchmark_name + " ",
            x=0.02,
            # y=0.70 if benchmark_name in ["Kalman Filter", "Sparse Attention"] else 1.99,
            y=0.70,
            xref=f"x{idx + 1}",
            yref=f"y{idx + 1}",
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font={"family": "serif", "size": 12, "weight": "bold", "color": "#333333"},
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=0.5,
            row=row,
            col=col,
        )

    fig.add_annotation(
        text="<b>Bandwidth</b>",
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.11,
        xanchor="center",
        yanchor="middle",
        showarrow=False,
        font={"family": "serif", "size": 12, "weight": "normal"},
    )

    fig.add_annotation(
        text="<b>Average Runtime (seconds) - Log Scale</b>",
        xref="paper",
        yref="paper",
        x=-0.08,
        y=0.5,
        textangle=-90,
        xanchor="center",
        yanchor="middle",
        showarrow=False,
        font={"family": "serif", "size": 12, "weight": "normal"},
    )

    fig.update_layout(
        title={
            "text": figure_title if show_title else None,
            "font": {"family": "serif", "size": 12, "weight": "normal"},
            "x": 0.5,
            "xanchor": "center",
        },
        showlegend=True,
        legend1={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.22,
            "xanchor": "center",
            "x": 0.5,
            "font": {"family": "serif", "size": 12},
            "bgcolor": "rgba(255, 255, 255, 0.95)",
        },
        legend2={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.30,
            "xanchor": "center",
            "x": 0.5,
            "entrywidthmode": "pixels",
            "entrywidth": 0,
            "font": {"family": "serif", "size": 12},
            "bgcolor": "rgba(255, 255, 255, 0.95)",
        },
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="serif",
        ),
        width=900,
        height=700,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=0, b=0, l=40, r=20),
    )

    output_path = f"./results/{output_prefix}"
    fig.write_html(f"{output_path}.html")

    if save_png_and_svg:
        fig.write_image(f"{output_path}.svg", width=550, height=410)
        fig.write_image(f"{output_path}.png", width=550, height=410, scale=2)

    fig.show()
    return fig


if __name__ == "__main__":
    benchmark_files = [
        "./results/kalman_filter.csv",
        # "./results/kalman_filter_dia_input.csv",
        "./results/sparse_attention.csv",
        # "./results/sparse_attention_dia_input.csv",
        "./results/batch_bertlike.csv",
        # "./results/batch_bertlike_dia_input.csv",
        "./results/chain.csv",
        # "./results/chain_dia_input.csv",
    ]

    fig = create_grouped_memory_plot(
        csv_paths=benchmark_files,
        output_prefix="memory_comparison",
        figure_title="Memory Footprint Across Benchmarks",
        show_title=False,
        save_png_and_svg=True,
    )
    fig = create_grouped_runtime_plot(
        csv_paths=benchmark_files,
        output_prefix="runtime_comparison",
        figure_title="Speedup Over Baseline",
        show_title=False,
        save_png_and_svg=True,
    )

    # fig = create_comptime_plot(
    #     csv_path="./results/compilation_time.csv",
    #     output_prefix="comp_time",
    #     show_title=False,
    #     save_png_and_svg=True,
    # )
