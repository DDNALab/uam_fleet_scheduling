"""
Scenario reduction.py
=====================
Backward scenario reduction with eVTOL filtering.

Uses Euclidean-distance-based backward elimination
on joint passenger/eVTOL time-series feature vectors.
"""

import numpy as np

from .config import TARGET_SCENARIOS


# ============================================================
# 1. Feature construction
# ============================================================
def build_feature_vectors(scenarios):
    """
    Attach feature vectors to scenarios:
    concatenation of passenger + eVTOL time series.
    """

    for s in scenarios:
        s["feature_vector"] = np.concatenate([
            np.asarray(s["a_t"]),
            np.asarray(s["e_t"])
        ])
    return scenarios


# ============================================================
# 2. eVTOL feasibility filter
# ============================================================
def filter_valid_scenarios(scenarios):
    """
    Keep only scenarios with at least one eVTOL event.
    """

    valid = [s for s in scenarios if s["total_evtols"] > 0]

    # fallback if too restrictive
    if len(valid) == 0:
        return sorted(scenarios, key=lambda x: x["total_evtols"], reverse=True)

    return valid


# ============================================================
# 3. Pairwise distance matrix
# ============================================================
def compute_distance_matrix(scenarios):
    """
    Euclidean distance between scenario feature vectors.
    """

    n = len(scenarios)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(
                scenarios[i]["feature_vector"] -
                scenarios[j]["feature_vector"]
            )
            dist[i, j] = dist[j, i] = d

    return dist


# ============================================================
# 4. Backward reduction core
# ============================================================
def backward_reduce(scenarios, target_size):
    """
    Iteratively remove least representative scenarios.
    """

    working = scenarios.copy()
    n0 = len(working)

    # initialize uniform probabilities
    for s in working:
        s["probability"] = 1.0 / n0

    while len(working) > target_size:

        dist = compute_distance_matrix(working)
        n = len(working)

        worst_cost = float("inf")
        remove_idx = None
        transfer_idx = None

        for i in range(n):

            mask = np.ones(n, dtype=bool)
            mask[i] = False

            min_dist = np.min(dist[i][mask])
            cost = working[i]["probability"] * min_dist

            if cost < worst_cost:
                worst_cost = cost
                remove_idx = i
                transfer_idx = np.argmin([
                    dist[i][j] if j != i else float("inf")
                    for j in range(n)
                ])

        # probability transfer
        working[transfer_idx]["probability"] += working[remove_idx]["probability"]

        # remove weakest scenario
        working.pop(remove_idx)

    return working


# ============================================================
# 5. Normalize probabilities
# ============================================================
def normalize_probabilities(scenarios):
    """
    Ensure probabilities sum to 1.
    """

    total = sum(s["probability"] for s in scenarios)

    if total > 0:
        for s in scenarios:
            s["probability"] /= total

    return scenarios


# ============================================================
# 6. Main reduction function
# ============================================================
def scenario_reduction_backward_with_evtol_filter(
        scenarios,
        target_size=TARGET_SCENARIOS,
        plot=True):

    """
    Backward scenario reduction with eVTOL constraint.
    """

    # 1. filter valid scenarios
    scenarios = filter_valid_scenarios(scenarios)

    # 2. feature vectors
    scenarios = build_feature_vectors(scenarios)

    # 3. reduction
    reduced = backward_reduce(scenarios, target_size)

    # 4. normalize
    reduced = normalize_probabilities(reduced)

    # 5. visualization
    if plot:
        plot_reduction(reduced, scenarios)

    return reduced


# ============================================================
# 7. Visualization (matplotlib is optional / lazy-loaded)
# ============================================================
def plot_reduction(reduced, original):
    """
    Lazily imports matplotlib only when a plot is actually requested.
    If matplotlib/Pillow is broken in the environment, this prints a
    warning and falls back to a console summary instead of crashing
    the whole script.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print(f"[warning] matplotlib unavailable ({e}); skipping plot.")
        _text_summary(reduced, original)
        return

    l_periods = len(original[0]["a_t"])
    t = np.arange(l_periods)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Scenario Reduction Results")

    colors = ["blue", "green", "red", "purple", "orange"]

    # passenger
    for i, s in enumerate(reduced):
        axs[0, 0].plot(t, s["a_t"],
                       label=f"S{s['id']} p={s['probability']:.2f}",
                       color=colors[i % len(colors)])
    axs[0, 0].set_title("Passenger arrivals")

    # eVTOL
    for i, s in enumerate(reduced):
        axs[0, 1].plot(t, s["e_t"],
                       label=f"S{s['id']} p={s['probability']:.2f}",
                       color=colors[i % len(colors)])
    axs[0, 1].set_title("eVTOL arrivals")

    # probabilities
    axs[1, 0].bar(
        [f"S{s['id']}" for s in reduced],
        [s["probability"] for s in reduced],
        color=colors[:len(reduced)]
    )
    axs[1, 0].set_title("Scenario probabilities")

    # summary comparison
    orig_p = np.mean([s["total_passengers"] for s in original])
    red_p = np.mean([s["total_passengers"] for s in reduced])

    orig_e = np.mean([s["total_evtols"] for s in original])
    red_e = np.mean([s["total_evtols"] for s in reduced])

    axs[1, 1].bar(
        ["Orig P", "Red P", "Orig E", "Red E"],
        [orig_p, red_p, orig_e, red_e]
    )
    axs[1, 1].set_title("Aggregate comparison")

    plt.tight_layout()
    plt.show()


def _text_summary(reduced, original):
    """Console-only fallback summary, no plotting libs required."""

    print("\nScenario Reduction Results (text summary)")
    print("-" * 45)
    for s in reduced:
        print(f"  S{s['id']:<4} probability={s['probability']:.3f}  "
              f"passengers={s['total_passengers']:<5} evtols={s['total_evtols']}")

    orig_p = np.mean([s["total_passengers"] for s in original])
    red_p = np.mean([s["total_passengers"] for s in reduced])
    orig_e = np.mean([s["total_evtols"] for s in original])
    red_e = np.mean([s["total_evtols"] for s in reduced])

    print("-" * 45)
    print(f"  Avg passengers  -> original: {orig_p:.1f}, reduced: {red_p:.1f}")
    print(f"  Avg eVTOLs      -> original: {orig_e:.1f}, reduced: {red_e:.1f}")
