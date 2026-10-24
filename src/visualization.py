"""
Plots for the scenario ensemble and a solved policy.

All figures are written as standalone HTML into ``config.FIGURES_DIR``. Plotly
is imported lazily so the rest of the package keeps working where only the
optimization stack is installed.

Figures
-------
1. on-demand demand trajectories by destination
2. arrivals and the arrival-SoC distribution
3. the shared temporal factor ``g_ts``
4. scenario weights before and after reduction, with ``N_eff``
5. charger occupancy from a solved model
6. DFW flow map (needs folium and the vertiport GeoDataFrame)
"""

import numpy as np

import config
def _plotly():
    """Import plotly lazily; return ``None`` when it is unavailable."""
    try:
        import plotly.graph_objects as go

        return go
    except Exception:
        print("[warning] plotly unavailable; skipping figure")
        return None


def _save(figure, filename):
    """
    Write one figure, reading ``config.FIGURES_DIR`` at call time so a caller can
    redirect output. A read-only output directory must not abort a run, so the
    failure is reported and ``None`` is returned.
    """
    directory = config.FIGURES_DIR

    path = directory / filename

    try:

        directory.mkdir(parents=True, exist_ok=True)

        figure.write_html(path, include_plotlyjs="cdn")

    except OSError as error:

        print("[warning] could not write %s (%s); skipping figure." % (path, error))

        return None

    print("  >> Saved: %s" % path)

    return path


# ============================================================
# 1. Realized on-demand demand
# ============================================================

def plot_ondemand_trajectories(scenarios, max_scenarios=12, filename="1_ondemand_trajectories.html"):
    """
    Realized on-demand requests per period, one line per scenario, one panel per
    destination.

    These series are part of the feature vector used for scenario reduction.
    """
    go = _plotly()

    if go is None or not scenarios:
        return None

    from plotly.subplots import make_subplots

    destinations = sorted({d for s in scenarios for (_, d) in s.ondemand_demand})

    periods = sorted({t for s in scenarios for (t, _) in s.ondemand_demand})

    figure = make_subplots(
        rows=len(destinations),
        cols=1,
        shared_xaxes=True,
        subplot_titles=destinations,
    )

    for row, destination in enumerate(destinations, start=1):

        for scenario in scenarios[:max_scenarios]:

            values = [
                float(scenario.ondemand_demand.get((t, destination), 0))
                for t in periods
            ]

            figure.add_trace(
                go.Scatter(
                    x=periods,
                    y=values,
                    mode="lines",
                    name="S%d (p=%.3f)" % (scenario.id, scenario.probability),
                    showlegend=(row == 1),
                ),
                row=row,
                col=1,
            )

    figure.update_layout(
        title="Realized on-demand demand by destination (Section 5.1)",
        height=260 * len(destinations) + 120,
        hovermode="x unified",
    )

    figure.update_xaxes(title_text="Period")

    return _save(figure, filename)


# ============================================================
# 2. Arrivals and arrival SoC
# ============================================================

def plot_arrivals_and_soc(scenarios, filename="2_arrivals_and_soc.html"):
    """
    Incoming arrivals per period and the arrival-SoC histogram.

    Arrival SoC is conditional on demand stress through the common factor, so the
    histogram shows the coupling; ``beta_Gamma = beta_g = 0`` gives a noticeably
    different shape.
    """
    go = _plotly()

    if go is None or not scenarios:
        return None

    from plotly.subplots import make_subplots

    periods = sorted({t for s in scenarios for (t, _) in s.ondemand_demand})

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Incoming aircraft arrivals", "Arrival SoC (Eq. 63)"),
    )

    for scenario in scenarios:

        counts = (
            scenario.arrival_counts
            if scenario.arrival_counts is not None
            else np.zeros(len(periods))
        )

        figure.add_trace(
            go.Bar(x=periods, y=list(counts), name="S%d" % scenario.id, showlegend=False),
            row=1,
            col=1,
        )

    socs = [a.initial_soc for s in scenarios for a in s.aircraft]

    if socs:
        figure.add_trace(
            go.Histogram(x=socs, nbinsx=20, name="SoC"),
            row=1,
            col=2,
        )

    figure.update_layout(
        title="Incoming supply and arrival state of charge",
        height=460,
        bargap=0.05,
    )

    figure.update_xaxes(title_text="Period", row=1, col=1)
    figure.update_xaxes(title_text="Arrival SoC (%)", row=1, col=2)

    return _save(figure, filename)


# ============================================================
# 3. Common factor
# ============================================================

def plot_common_factor(scenarios, max_scenarios=12, filename="3_common_factor.html"):
    """
    The shared temporal factor ``g_ts`` that couples on-demand demand and
    incoming supply.
    """
    go = _plotly()

    if go is None or not scenarios:
        return None

    figure = go.Figure()

    for scenario in scenarios[:max_scenarios]:

        factor = np.asarray(scenario.common_factor, dtype=float)

        figure.add_trace(
            go.Scatter(
                x=list(range(1, len(factor) + 1)),
                y=list(factor),
                mode="lines",
                name="S%d" % scenario.id,
            )
        )

    figure.update_layout(
        title="Shared temporal factor g<sub>ts</sub>",
        xaxis_title="Period",
        yaxis_title="Standardized factor",
        height=420,
        hovermode="x unified",
    )

    return _save(figure, filename)


# ============================================================
# 4. Reduction diagnosis
# ============================================================

def plot_reduction_diagnostics(raw, reduced, filename="4_reduction.html"):
    """
    Proxy scores, retained weights, and the effective scenario count before and
    after reduction.

    Tail stratification does not by itself guarantee an accurate CVaR estimate,
    so check the weight profile and ``N_eff`` first.
    """
    go = _plotly()

    if go is None or not reduced:
        return None

    from plotly.subplots import make_subplots

    from scenario_reduction import effective_scenario_count

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Proxy score psi_s (Eq. 65)", "Retained weights"),
    )

    if raw:
        scores = [
            scenario.proxy_score
            if scenario.proxy_score is not None
            else float("nan")
            for scenario in raw
        ]

        figure.add_trace(
            go.Scatter(
                x=[s.id for s in raw],
                y=scores,
                mode="markers",
                marker=dict(
                    color=["crimson" if s.stratum == "tail" else "steelblue" for s in raw],
                    size=7,
                ),
                name="raw",
            ),
            row=1,
            col=1,
        )

    figure.add_trace(
        go.Bar(
            x=["S%d" % s.id for s in reduced],
            y=[float(s.probability) for s in reduced],
            marker_color=[
                "crimson" if s.stratum == "tail" else "steelblue" for s in reduced
            ],
            name="retained",
        ),
        row=1,
        col=2,
    )

    raw_effective = (
        effective_scenario_count(np.full(len(raw), 1.0 / len(raw))) if raw else 0.0
    )

    reduced_effective = effective_scenario_count(
        np.array([float(s.probability) for s in reduced])
    )

    figure.update_layout(
        title=(
            "Scenario reduction: N_eff %.2f -> %.2f (tail %d of %d)"
            % (
                raw_effective,
                reduced_effective,
                sum(1 for s in reduced if s.stratum == "tail"),
                len(reduced),
            )
        ),
        height=460,
        showlegend=False,
    )

    return _save(figure, filename)


# ============================================================
# 5. Charger occupancy from a solved model
# ============================================================

def plot_charger_occupancy(model_data, filename="5_charger_occupancy.html"):
    """Charger occupancy per period against the reserved capacity ``b_t``."""
    go = _plotly()

    if go is None:
        return None

    from plotly.subplots import make_subplots

    from solution_export import charger_occupancy

    variables = model_data["variables"]
    periods = [int(t) for t in model_data["periods"]]
    facilities = float(model_data["parameters"]["charging_facilities"])

    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Charger occupancy", "Reserved capacity b<sub>t</sub>"),
    )

    for scenario_id in model_data["scenarios"]:

        occupancy = charger_occupancy(scenario_id, model_data)

        figure.add_trace(
            go.Scatter(
                x=periods,
                y=[occupancy[t] for t in periods],
                mode="lines",
                name="S%d" % scenario_id,
            ),
            row=1,
            col=1,
        )

    figure.add_trace(
        go.Bar(
            x=periods,
            y=[float(variables["b"][t].value() or 0.0) for t in periods],
            name="b_t",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    figure.add_hline(
        y=facilities, line_dash="dash", line_color="black", row=1, col=2
    )

    figure.update_layout(
        title="Charging capacity: realized occupancy versus reservation",
        height=440,
        hovermode="x unified",
    )

    return _save(figure, filename)


# ============================================================
# 6. Flow map (optional)
# ============================================================

def plot_flow_map(verts_gdf, flows_df, threshold=50.0, filename="6_flow_map.html"):
    """
    DFW hub flow map.

    Needs ``folium`` and the vertiport GeoDataFrame, so it is the one figure
    that is skipped in a bare environment.
    """
    try:
        import folium
    except ImportError as error:
        print("[warning] folium unavailable (%s); skipping flow map." % error)
        return None

    if flows_df is None or verts_gdf is None:
        print("[warning] no gravity table available; skipping flow map.")
        return None

    network = folium.Map(location=[32.85, -97.0], zoom_start=10)

    for _, row in verts_gdf.iterrows():

        folium.Marker(
            location=[row.geometry.y, row.geometry.x],
            tooltip=row["name"],
            icon=folium.Icon(color="blue", icon="plane", prefix="fa"),
        ).add_to(network)

    for _, row in flows_df.iterrows():

        trips = float(row.get("expected_trips", row.get("trips", 0.0)))

        if trips < threshold:
            continue

        origin = verts_gdf[verts_gdf["name"] == row["origin"]].geometry.iloc[0]

        destination = verts_gdf[verts_gdf["name"] == row["destination"]].geometry.iloc[0]

        folium.PolyLine(
            locations=[[origin.y, origin.x], [destination.y, destination.x]],
            weight=min(15.0, trips / 100.0),
            color="black",
            opacity=0.6,
            tooltip="%s -> %s (%.1f)" % (row["origin"], row["destination"], trips),
        ).add_to(network)

    directory = config.FIGURES_DIR

    path = directory / filename

    try:

        directory.mkdir(parents=True, exist_ok=True)

        network.save(path)

    except OSError as error:

        print("[warning] could not write %s (%s); skipping figure." % (path, error))

        return None

    print("  >> Saved: %s" % path)

    return network


# ============================================================
# 7. Console summary
# ============================================================

def summarize_solution(summary):
    """Compact console summary built from :func:`solution_export.build_summary`."""
    line = "=" * 60

    print("\n" + line)
    print(" SOLUTION SUMMARY")
    print(line)

    for key in (
        "formulation",
        "scenarios",
        "x0",
        "planned_departures",
        "expected_profit",
        "cvar",
        "cvar_penalty",
        "objective",
    ):
        if key in summary:
            value = summary[key]
            print(
                " %-22s %s"
                % (key, ("%.4f" % value) if isinstance(value, float) else value)
            )

    service = summary.get("service")

    if service:
        print(
            " booked service rate    %.4f"
            % service.get("booked_service_rate", float("nan"))
        )
        print(
            " on-demand service rate %.4f"
            % service.get("ondemand_service_rate", float("nan"))
        )

    print(line)
