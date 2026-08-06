"""
Model construction.py
=====================
Stochastic optimization input builder + conditional SoC integration.

Converts reduced scenarios into solver-ready structure:
- eVTOL charging states
- passenger demand tensor (time × destination)
- scenario probabilities
"""

import numpy as np
from collections import defaultdict

from .config import (
    HORIZON_START, HORIZON_END, BIN_SIZE,
    SOC_MIN, BATTERY_CAPACITY, CHARGING_RATE,
    LOS, EVTOL_CAPACITY, M_FACILITIES
)
from .scenario_generation import energy_price_per_kwh


# ============================================================
# 1. SoC Sampling (conditional, per scenario)
# ============================================================
def conditional_soc_sampling_per_evtol(
        reduced_scenarios,
        assignment_case=None,
        s_base=50,
        k=12,
        sigma_base=3,
        sigma_scale=5,
        plot=False):
    """
    Assign stochastic SoC values to each eVTOL conditional on load ratio.
    """

    for scenario in reduced_scenarios:

        a_t = np.asarray(scenario["a_t"], dtype=float)
        e_t = np.asarray(scenario["e_t"], dtype=float)

        # --------------------------------------------------------
        # Load ratio (passenger pressure per available eVTOLs)
        # --------------------------------------------------------
        rho = np.divide(
            a_t,
            e_t,
            out=np.zeros_like(a_t, dtype=float),
            where=e_t > 0,
        )

        # Carry the latest defined load ratio through bins without an arrival.
        # A leading empty bin has no meaningful ratio and remains zero.
        for t in range(1, len(rho)):
            if e_t[t] == 0:
                rho[t] = rho[t - 1]

        evtol_socs = {}
        evtol_arrival_bins = {}

        # --------------------------------------------------------
        # Case-based adjustment
        # --------------------------------------------------------
        if assignment_case == 1:
            s_base_adj = s_base + 2
            k_adj = k
        elif assignment_case == 2:
            s_base_adj = s_base - 3
            k_adj = k * 1.1
        else:
            s_base_adj = s_base
            k_adj = k

        evtol_id = 0

        # --------------------------------------------------------
        # Sampling loop
        # --------------------------------------------------------
        for t, cnt in enumerate(e_t):

            if cnt <= 0:
                continue

            mu = s_base_adj - k_adj * rho[t]
            sigma = sigma_base + sigma_scale * rho[t]

            mu = np.clip(mu, 25, 55)
            sigma = np.clip(sigma, 2, 8)

            for _ in range(int(cnt)):

                soc = None
                while soc is None or not (20 <= soc <= 60):
                    soc = np.random.normal(mu, sigma)

                evtol_socs[evtol_id] = float(soc)
                evtol_arrival_bins[evtol_id] = int(t + 1)

                evtol_id += 1

        scenario["rho_t"] = rho
        scenario["evtol_socs"] = evtol_socs
        scenario["evtol_arrival_periods"] = evtol_arrival_bins
        scenario["total_evtols_count"] = evtol_id
        scenario["avg_load_ratio"] = float(np.mean(rho))
        scenario["assignment_case"] = assignment_case

    return reduced_scenarios


# ============================================================
# 2. Stochastic Model Input Builder
# ============================================================
def build_stochastic_model_input(
        reduced_scenarios,
        bin_pmfs,
        destination_fares,
        destination_mapping,
        m_facilities=M_FACILITIES,
        l_periods=None,
        soc_min=SOC_MIN,
        c=BATTERY_CAPACITY,
        qr=CHARGING_RATE,
        los=LOS,
        eVTOL_capacity=EVTOL_CAPACITY,
        verbose=True):

    """
    Convert reduced scenarios into solver-ready stochastic input.
    """

    if l_periods is None:
        l_periods = (HORIZON_END - HORIZON_START) // BIN_SIZE

    id_to_name = destination_mapping["id_to_name"]
    name_to_id = destination_mapping["name_to_id"]

    model = {
        "scenarios": {},
        "global_parameters": {
            "m_facilities": m_facilities,
            "l_periods": l_periods,
            "SoC_min": soc_min,
            "c": c,
            "qr": qr,
            "LOS": los,
            "eVTOL_capacity": eVTOL_capacity,
            "horizon": {
                "start": HORIZON_START,
                "end": HORIZON_END,
                "bin_size": BIN_SIZE
            },
            "destination_fares": destination_fares,
            "destination_mapping": destination_mapping
        }
    }

    # ============================================================
    # 3. Scenario Loop
    # ============================================================
    for scenario in reduced_scenarios:

        sid = int(scenario["id"])
        prob = float(scenario["probability"])

        a_t = np.asarray(scenario["a_t"], dtype=int)
        e_t = np.asarray(scenario["e_t"], dtype=int)

        if len(a_t) != l_periods:
            raise ValueError(f"Scenario {sid}: a_t length mismatch")

        if len(e_t) != l_periods:
            raise ValueError(f"Scenario {sid}: e_t length mismatch")

        scenario_block = {
            "probability": prob,
            "n_evtols": int(scenario.get("total_evtols_count", e_t.sum())),
            "w_passengers": int(a_t.sum()),
            "evtols": {},
            "agg_pass_demand": {}
        }

        # --------------------------------------------------------
        # 3.1 eVTOL processing
        # --------------------------------------------------------
        arrival_bins = list(scenario.get("evtol_arrival_periods", {}).values())
        soc_map = scenario.get("evtol_socs", {})

        evtol_list = []
        for i, b in enumerate(arrival_bins):
            soc = float(soc_map.get(i, np.random.uniform(20, 60)))
            evtol_list.append((i, b, soc))

        for j, (eid, bin_t, soc) in enumerate(evtol_list):

            if soc < soc_min:
                energy_needed = c * ((soc_min - soc) / 100)
                hours = energy_needed / qr
                periods = int(np.ceil(hours * (60 / BIN_SIZE)))

                finish = min(bin_t + periods, l_periods)

                cost = 0.0
                t_min = HORIZON_START + (bin_t - 1) * BIN_SIZE

                for _ in range(periods):
                    hour = (t_min % 1440) / 60.0
                    cost += (qr / (60 / BIN_SIZE)) * energy_price_per_kwh(hour)
                    t_min += BIN_SIZE
            else:
                periods = 0
                finish = bin_t
                cost = 0.0

            scenario_block["evtols"][j] = {
                "arrival_bin": bin_t,
                "soc": soc,
                "charging_periods": periods,
                "finish_charging_period": finish,
                "charging_cost": cost
            }

        # --------------------------------------------------------
        # 3.2 Passenger demand (bin × destination)
        # --------------------------------------------------------
        pmf_ids = []

        for t in range(l_periods):
            pmf = bin_pmfs[t]

            names = []
            probs = []

            for n, p in pmf.items():
                if n in name_to_id:
                    names.append(n)
                    probs.append(p)

            if len(names) == 0:
                ids = list(destination_fares.keys())
                probs = np.ones(len(ids)) / len(ids)
            else:
                ids = [name_to_id[n] for n in names]
                probs = np.array(probs)
                probs = probs / probs.sum()

            pmf_ids.append((ids, probs))

        demand = defaultdict(int)

        for t, cnt in enumerate(a_t, start=1):

            if cnt <= 0:
                continue

            ids, probs = pmf_ids[t - 1]
            samples = np.random.choice(ids, size=int(cnt), p=probs)

            for d in samples:
                demand[(t, int(d))] += 1

        # ensure full grid
        for t in range(1, l_periods + 1):
            for d in destination_fares.keys():
                demand.setdefault((t, int(d)), 0)

        scenario_block["agg_pass_demand"] = dict(demand)
        model["scenarios"][sid] = scenario_block

    # ============================================================
    # 4. Global summary
    # ============================================================
    ev = [s["n_evtols"] for s in model["scenarios"].values()]
    ps = [s["w_passengers"] for s in model["scenarios"].values()]

    model["global_parameters"]["n_evtols_range"] = (min(ev), max(ev))
    model["global_parameters"]["w_passengers_range"] = (min(ps), max(ps))

    # ============================================================
    # 5. Print summary
    # ============================================================
    if verbose:
        print("\n=== STOCHASTIC MODEL INPUT ===")
        print(f"Scenarios: {len(model['scenarios'])}")
        print(f"Time bins: {l_periods} ({BIN_SIZE} min)")

    return model
