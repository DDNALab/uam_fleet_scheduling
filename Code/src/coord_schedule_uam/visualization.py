"""
Visualization.py
================
Post-processing and diagnostics for:
- demand scenarios
- reduction quality
- stochastic solution outputs
Using plotly instead of matplotlib.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import FIGURES_DIR

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def _save_and_show(fig, filename):
    path = FIGURES_DIR / filename
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"  >> Saved: {path}")


# ============================================================
# 1. Scenario arrival diagnostics (Image 1 style - histogram)
# ============================================================
def plot_scenario_arrivals(scenarios, title="Scenario Arrival Profiles"):
    fig = make_subplots(rows=2, cols=1, subplot_titles=[
        "Passenger Arrivals (DFW scenario) — bimodal demand",
        "eVTOL Arrivals (DFW scenario) — smooth bimodal"
    ])
    for s in scenarios:
        t = list(range(len(s["a_t"])))
        fig.add_trace(go.Bar(x=t, y=s["a_t"].tolist(), name=f"S{s['id']} p={s['probability']:.2f}",
                             marker_color="steelblue"), row=1, col=1)
        fig.add_trace(go.Bar(x=t, y=s["e_t"].tolist(), name=f"S{s['id']}",
                             marker_color="steelblue"), row=2, col=1)
    fig.update_layout(title=title, height=600, showlegend=False)
    fig.update_xaxes(title_text="Time (min since midnight)")
    fig.update_yaxes(title_text="Count")
    _save_and_show(fig, "1_scenario_arrivals.html")



# ============================================================
# 2. Scenario probability distribution
# ============================================================
def plot_scenario_probabilities(scenarios):
    colors = ["blue", "green", "red", "purple", "orange"]
    labels = [f"Scenario {s['id']}" for s in scenarios]
    probs  = [s["probability"] for s in scenarios]
    fig = go.Figure([
        go.Bar(x=[labels[i]], y=[probs[i]],
               name=labels[i],
               marker_color=colors[i % len(colors)],
               text=[f"{probs[i]:.2f}"], textposition="outside")
        for i in range(len(scenarios))
    ])
    fig.update_layout(
        title="Scenario Probabilities",
        xaxis_title="Scenario",
        yaxis_title="Probability",
        showlegend=False
    )
    _save_and_show(fig, "2_scenario_probabilities.html")



# ============================================================
# 3. Aggregate demand comparison
# ============================================================
def plot_demand_comparison(original, reduced):
    orig_p = np.mean([s["total_passengers"] for s in original])
    red_p  = np.mean([s["total_passengers"] for s in reduced])
    orig_e = np.mean([s["total_evtols"] for s in original])
    red_e  = np.mean([s["total_evtols"] for s in reduced])
    fig = go.Figure([
        go.Bar(name="Original Avg", x=["Passengers", "eVTOLs"], y=[orig_p, orig_e],
               marker_color="steelblue"),
        go.Bar(name="Reduced Avg",  x=["Passengers", "eVTOLs"], y=[red_p, red_e],
               marker_color="orange")
    ])
    fig.update_layout(
        title="Original vs Reduced Scenarios",
        barmode="group",
        yaxis_title="Average Count"
    )
    _save_and_show(fig, "3_demand_comparison.html")



# ============================================================
# 4. 4-panel reduced scenario analysis (Image 2 style)
# ============================================================
def plot_reduced_scenario_analysis(scenarios, original):
    colors = ["blue", "green", "red", "purple", "orange"]

    fig = make_subplots(rows=2, cols=2, subplot_titles=[
        "Passenger Arrivals per Time Period",
        "eVTOL Arrivals per Time Period",
        "Scenario Probabilities",
        "Original vs Reduced Scenarios"
    ])

    # --- Passenger + eVTOL lines ---
    for i, s in enumerate(scenarios):
        t = list(range(1, len(s["a_t"]) + 1))
        c = colors[i % len(colors)]
        label = f"Scenario {s['id']} (p={s['probability']:.2f})"
        fig.add_trace(go.Scatter(
            x=t, y=s["a_t"].tolist(), name=label,
            line=dict(color=c), mode="lines+markers"
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=t, y=s["e_t"].tolist(), name=label,
            line=dict(color=c), mode="lines+markers",
            showlegend=False
        ), row=1, col=2)

    # --- Probabilities bar ---
    for i, s in enumerate(scenarios):
        fig.add_trace(go.Bar(
            x=[f"Scenario {s['id']}"],
            y=[s["probability"]],
            marker_color=colors[i % len(colors)],
            text=[f"{s['probability']:.2f}"],
            textposition="outside",
            showlegend=False
        ), row=2, col=1)

    # --- Original vs Reduced ---
    orig_p = np.mean([s["total_passengers"] for s in original])
    red_p  = np.mean([s["total_passengers"] for s in scenarios])
    orig_e = np.mean([s["total_evtols"] for s in original])
    red_e  = np.mean([s["total_evtols"] for s in scenarios])

    fig.add_trace(go.Bar(
        name="Original Avg",
        x=["Passengers", "eVTOLs"],
        y=[orig_p, orig_e],
        marker_color="steelblue"
    ), row=2, col=2)

    fig.add_trace(go.Bar(
        name="Reduced Avg",
        x=["Passengers", "eVTOLs"],
        y=[red_p, red_e],
        marker_color="orange"
    ), row=2, col=2)

    fig.update_layout(
        title="<b>Reduced Scenario Set Analysis</b>",
        height=800,
        barmode="group"
    )
    fig.update_xaxes(title_text="Time Period (10-min bins)", row=1, col=1)
    fig.update_xaxes(title_text="Time Period (10-min bins)", row=1, col=2)
    fig.update_yaxes(title_text="Passenger Count", row=1, col=1)
    fig.update_yaxes(title_text="eVTOL Count", row=1, col=2)
    fig.update_yaxes(title_text="Probability", row=2, col=1)
    fig.update_yaxes(title_text="Average Count", row=2, col=2)
    _save_and_show(fig, "4_reduced_scenario_analysis.html")



# ============================================================
# 5. eVTOL SoC distribution
# ============================================================
def plot_soc_distribution(reduced_scenarios):
    socs = [
        soc
        for s in reduced_scenarios
        if "evtol_socs" in s
        for soc in s["evtol_socs"].values()
    ]
    fig = go.Figure(go.Histogram(x=socs, nbinsx=10, marker_color="steelblue"))
    fig.update_layout(
        title="eVTOL State of Charge Distribution",
        xaxis_title="SoC (%)",
        yaxis_title="Frequency"
    )
    _save_and_show(fig, "5_soc_distribution.html")



# ============================================================
# 6. Flow visualization (DFW hub network)
# ============================================================
def plot_flow_map(verts_gdf, flows_df, threshold=50):
    try:
        import folium
    except ImportError as e:
        print(f"[warning] folium unavailable ({e}); skipping flow map.")
        return None

    m = folium.Map(location=[32.85, -97.0], zoom_start=10)

    for _, r in verts_gdf.iterrows():
        folium.Marker(
            location=[r.geometry.y, r.geometry.x],
            tooltip=r["name"],
            icon=folium.Icon(color="blue", icon="plane", prefix="fa")
        ).add_to(m)

    for _, row in flows_df.iterrows():
        trips = float(row.get("trips", row.get("DFW_to_dest_trips", 0)))
        if trips < threshold:
            continue
        o = verts_gdf[verts_gdf["name"] == row["origin"]].geometry.iloc[0]
        d = verts_gdf[verts_gdf["name"] == row["destination"]].geometry.iloc[0]
        folium.PolyLine(
            locations=[[o.y, o.x], [d.y, d.x]],
            weight=min(15, trips / 100),
            color="black",
            opacity=0.6,
            tooltip=f"{row['origin']} → {row['destination']} ({trips:.1f})"
        ).add_to(m)
    path = FIGURES_DIR / "6_flow_map.html"
    m.save(path)
    print(f"  >> Saved: {path}")
    return m


# ============================================================
# 7. Optimization solution summary
# ============================================================
def summarize_solution(model, results):
    print("\n==============================")
    print(" SOLUTION SUMMARY")
    print("==============================")
    print(f"Status    : {results.get('status')}")
    print(f"Objective : {results.get('objective')}")
    if "CVaR" in results:
        print(f"CVaR      : {results['CVaR']:.2f}")
        print(f"VaR       : {results['VaR_zeta']:.2f}")
        print(f"Expected U: {results['Expected_Unserved']:.2f}")
        print(f"Risk λ    : {results['Risk_Weight']}")
    print("==============================")
