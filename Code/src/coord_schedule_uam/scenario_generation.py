"""
Scenario generation.py
======================
NHPP-based passenger/eVTOL arrivals + destination PMFs
+ charging simulation + energy pricing.
"""

import numpy as np
import random

from collections import Counter
from scipy.ndimage import gaussian_filter1d

from .config import (
    BASE_PRICE, M_PEAK, M_SHOULDER, M_OFF,
    SOC_MIN, BATTERY_CAPACITY, CHARGING_RATE,
    HORIZON_START, HORIZON_END, BIN_SIZE,
    N_SCENARIOS, SEED
)

# ============================================================
# 1. Time-of-Use Energy Pricing
# ============================================================
def energy_price_per_kwh(t_hour,
                         base=BASE_PRICE,
                         m_peak=M_PEAK,
                         m_shoulder=M_SHOULDER,
                         m_off=M_OFF):
    """
    TOU electricity pricing.
    t_hour in [0, 24).
    """
    if not (0 <= t_hour < 24):
        raise ValueError("t must be in [0,24)")

    if 16 <= t_hour < 21:
        return base * m_peak
    elif 0 <= t_hour < 6 or 21 <= t_hour < 24:
        return base * m_off
    else:
        return base * m_shoulder


# ============================================================
# 2. eVTOL Charging + Passenger Execution Simulation
# ============================================================
def simulate_evtol_and_passengers(scenario_data,
                                  seed=SEED,
                                  plot=True):
    """
    Simulates:
    - eVTOL SoC depletion/charging cost
    - passenger fares (no resampling)
    """

    np.random.seed(seed)
    random.seed(seed)

    destination_data = scenario_data["destinations"]
    passenger_arrivals = np.asarray(scenario_data["passenger_arrivals"])
    passenger_dest = np.asarray(scenario_data["passenger_destinations"])
    evtol_arrivals = np.asarray(scenario_data["evtol_arrivals"])

    # -----------------------------
    # Initial SoC
    # -----------------------------
    SoC = {
        j: random.uniform(20, 60)
        for j in range(len(evtol_arrivals))
    }

    charging_cost = {}
    charging_periods = {}
    finish_time = {}

    # -----------------------------
    # Charging process
    # -----------------------------
    for j, arrival_min in enumerate(evtol_arrivals):

        if SoC[j] < SOC_MIN:

            energy_needed = BATTERY_CAPACITY * ((SOC_MIN - SoC[j]) / 100)
            hours = energy_needed / CHARGING_RATE
            periods = int(np.ceil(hours * (60 / BIN_SIZE)))

            charging_periods[j] = periods

            cost = 0.0
            t = float(arrival_min)

            remaining = periods

            while remaining > 0 and t < HORIZON_END:

                price = energy_price_per_kwh(t / 60.0)
                energy = CHARGING_RATE * (BIN_SIZE / 60.0)

                cost += energy * price

                t += BIN_SIZE
                remaining -= 1

            charging_cost[j] = cost
            finish_time[j] = t

        else:
            charging_periods[j] = 0
            finish_time[j] = float(arrival_min)
            charging_cost[j] = 0.0

    # -----------------------------
    # Passenger fares
    # -----------------------------
    fares = {}
    for pid, d in enumerate(passenger_dest):
        fares[pid] = destination_data[d]["fare"]

    # -----------------------------
    # Visualization (matplotlib is optional / lazy-loaded)
    # -----------------------------
    if plot:
        _plot_arrivals(passenger_arrivals, evtol_arrivals)

    return {
        "SoC": SoC,
        "charging_cost": charging_cost,
        "charging_periods": charging_periods,
        "finish_time": finish_time,
        "passenger_destinations": passenger_dest,
        "passenger_fares": fares
    }


def _plot_arrivals(passenger_arrivals, evtol_arrivals):
    """
    Lazily imports matplotlib only when a plot is actually requested.
    If matplotlib/Pillow is broken in the environment, this prints a
    warning and falls back to a quick text histogram instead of
    crashing the whole script.
    """
    bins = np.arange(HORIZON_START, HORIZON_END + BIN_SIZE, BIN_SIZE)

    try:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        axs[0].hist(passenger_arrivals, bins=bins, edgecolor="black", alpha=0.7)
        axs[0].set_title("Passenger Arrivals")
        axs[0].set_ylabel("Count")

        axs[1].hist(evtol_arrivals, bins=bins, edgecolor="black", alpha=0.7)
        axs[1].set_title("eVTOL Arrivals")
        axs[1].set_ylabel("Count")
        axs[1].set_xlabel("Time (min)")

        plt.tight_layout()
        plt.show()

    except ImportError as e:
        print(f"[warning] matplotlib unavailable ({e}); skipping plot.")
        _text_histogram("Passenger Arrivals", passenger_arrivals, bins)
        _text_histogram("eVTOL Arrivals", evtol_arrivals, bins)


def _text_histogram(title, data, bins, width=40):
    """Simple console fallback histogram, no plotting libs required."""
    counts, _ = np.histogram(data, bins=bins)
    max_count = counts.max() if counts.size and counts.max() > 0 else 1

    print(f"\n{title}")
    for i, c in enumerate(counts):
        bar = "#" * int(width * c / max_count)
        print(f"  [{int(bins[i]):>4}-{int(bins[i+1]):<4}] {bar} ({c})")


# ============================================================
# 3. Bin-level Destination PMFs
# ============================================================
def build_bin_destination_pmfs(passenger_arrivals,
                               passenger_dest,
                               target_shares=None,
                               alpha=1.0,
                               blend_weight=0.15):

    passenger_arrivals = np.asarray(passenger_arrivals)
    dest_list = list(passenger_dest)

    l_periods = (HORIZON_END - HORIZON_START) // BIN_SIZE
    bins = np.arange(HORIZON_START, HORIZON_END + BIN_SIZE, BIN_SIZE)

    bin_idx = np.clip(
        ((passenger_arrivals - HORIZON_START) // BIN_SIZE).astype(int),
        0, l_periods - 1
    )

    per_bin = [Counter() for _ in range(l_periods)]
    all_dests = sorted(set(dest_list))

    for i, b in enumerate(bin_idx):
        per_bin[b][dest_list[i]] += 1

    global_counts = Counter(dest_list)
    total = sum(global_counts.values()) or 1
    global_p = {d: global_counts[d] / total for d in all_dests}

    if target_shares is None:
        target_shares = {}
        blend_weight = 0.0

    total_target = sum(target_shares.values()) or 1
    target_p = {d: target_shares.get(d, 0) / total_target for d in all_dests}

    pmfs = []

    for t in range(l_periods):
        counts = per_bin[t]
        N = sum(counts.values())

        pmf = {}

        for d in all_dests:
            prior = global_p.get(d, 1 / len(all_dests))
            pmf[d] = (counts.get(d, 0) + alpha * prior) / (N + alpha)

        if blend_weight > 0:
            for d in pmf:
                pmf[d] = (1 - blend_weight) * pmf[d] + blend_weight * target_p.get(d, 0)

        s = sum(pmf.values()) or 1e-9
        for d in pmf:
            pmf[d] /= s

        pmfs.append(pmf)

    return pmfs, all_dests


# ============================================================
# 4. NHPP Scenario Generator (core engine)
# ============================================================
def generate_joint_demand_scenarios(passenger_arrivals,
                                    evtol_arrivals,
                                    destination_data,
                                    bin_pmfs,
                                    n_scenarios=N_SCENARIOS):

    start, end = HORIZON_START, HORIZON_END
    l_periods = (end - start) // BIN_SIZE

    bins = np.linspace(start, end, l_periods + 1)

    # baseline histograms
    pax_hist, _ = np.histogram(passenger_arrivals, bins=bins)
    evt_hist, _ = np.histogram(evtol_arrivals, bins=bins)

    f_a = gaussian_filter1d(pax_hist.astype(float), sigma=2)
    f_e = gaussian_filter1d(evt_hist.astype(float), sigma=2)

    var = min(np.var(pax_hist - f_a), np.var(evt_hist - f_e))

    scenarios = []

    for s in range(n_scenarios):

        eps = np.random.normal(0, np.sqrt(var), l_periods)

        lambda_a = np.maximum(f_a + eps, 0)
        lambda_e = np.maximum(f_e + eps, 0)

        a_t = np.random.poisson(lambda_a)
        e_t = np.random.poisson(lambda_e)

        pid = 0
        dest_map = {}
        fare_map = {}

        for t in range(l_periods):

            n = int(a_t[t])
            if n == 0:
                continue

            pmf = bin_pmfs[t]
            dests = list(pmf.keys())
            probs = np.array(list(pmf.values()))
            probs = probs / probs.sum()

            choices = np.random.choice(dests, size=n, p=probs)

            for d in choices:
                dest_map[pid] = d
                fare_map[pid] = destination_data[d]["fare"]
                pid += 1

        scenarios.append({
            "id": s,
            "a_t": a_t,
            "e_t": e_t,
            "total_passengers": int(a_t.sum()),
            "total_evtols": int(e_t.sum()),
            "passenger_destinations": dest_map,
            "passenger_fares": fare_map,
            "destination_counts": dict(Counter(dest_map.values()))
        })

    return scenarios
