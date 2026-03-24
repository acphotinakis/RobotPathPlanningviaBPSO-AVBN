"""main.py — Robot Path Planning experiments with BPSO + AVBN.

Reproduces the analysis from Mo & Xu (2015) and adds four new
diagnostic plots:

  Plot 1  — Success-rate comparison  (Table 2 equivalent)
  Plot 2  — Box-and-whisker cost distribution  (Table 1 equivalent)
  Plot 3  — Population diversity over generations
  Plot 4  — Node-visit-frequency heatmap (AVBN search-space density)

Plus the original:
  Plot 0  — Convergence curves (Figure 6 equivalent)
"""

import copy
import os
import warnings
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm
from typing import List, Tuple, Dict

from robot_path_planner import RobotPathPlanner
from bpso import BPSO, _remove_from_all

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Shared constants — paper §4.2
# ---------------------------------------------------------------------------
N_TRIALS = 50  # paper runs 50–100 trials per algorithm
POP_SIZE = 100
MAX_GEN = 100
GRID = (400, 400)
START = (10, 10)
GOAL = (350, 350)
OUTPUT_DIR = "output"

ALGO_COLOURS = {
    "GA": "#E24B4A",
    "PSO": "#EF9F27",
    "BBO": "#378ADD",
    "BPSO": "#3B6D11",
}


# ---------------------------------------------------------------------------
# Test environments  (Figure 4)
# ---------------------------------------------------------------------------


def create_test_environments():
    env1 = [
        (20, 20, 15, 30),
        (80, 60, 40, 25),
        (150, 30, 30, 20),
        (200, 100, 25, 35),
        (100, 150, 20, 15),
        (250, 180, 30, 25),
        (180, 220, 25, 30),
        (50, 250, 20, 25),
        (300, 250, 35, 20),
    ]
    env2 = [
        (50, 50, 20, 80),
        (150, 30, 80, 20),
        (250, 50, 20, 80),
        (100, 150, 20, 80),
        (200, 180, 80, 20),
        (150, 250, 20, 50),
    ]
    env3 = [
        (50, 50, 30, 40),
        (50, 90, 20, 30),
        (70, 80, 30, 10),
        (150, 100, 50, 50),
        (170, 150, 40, 30),
        (190, 130, 30, 20),
        (250, 200, 40, 40),
        (100, 250, 40, 30),
    ]
    return env1, env2, env3


# ---------------------------------------------------------------------------
# Comparison algorithm runners
# ---------------------------------------------------------------------------


def _build_planner(obstacles):
    """Return a ready-to-use planner for the given obstacle list."""
    p = RobotPathPlanner(grid_size=GRID)
    p.setup_environment(obstacles)
    return p


def run_bpso(planner: RobotPathPlanner) -> Tuple[float, bool]:
    """Full BPSO as fixed in bpso.py."""
    path, cost, _ = planner.plan_path(START, GOAL, pop_size=POP_SIZE, max_gen=MAX_GEN)
    success = (path is not None) and (cost < 1000)
    return (cost if success else float("inf")), success


def run_bbo_only(planner: RobotPathPlanner) -> Tuple[float, bool]:
    """BBO without the PSO position-update fallback (pure BBO, §2.1).

    Implemented by using BPSO with the PSO branch disabled: we monkey-patch
    the update so the else-branch (PSO) does nothing, leaving only migration
    and mutation.
    """
    nodes = planner.avbn.nodes
    connections = planner.avbn.node_connections
    dist_matrix = planner.avbn.distance_matrix
    if not nodes:
        return float("inf"), False

    start_node = planner._find_nearest_node(START, nodes)
    end_node = planner._find_nearest_node(GOAL, nodes)

    bbo = BPSO(pop_size=POP_SIZE, max_gen=MAX_GEN)
    bbo.initialize_population(len(nodes))

    gbf = float("inf")
    gb = None

    for generation in range(MAX_GEN):
        bbo._w = bbo.w_max - (bbo.w_max - bbo.w_min) * generation / max(MAX_GEN - 1, 1)

        bbo.fitness = []
        for i, ind in enumerate(bbo.population):
            path = bbo.decode_path(ind, connections, start_node, end_node)
            fitness = bbo.calculate_fitness(path, dist_matrix, end_node)
            bbo.fitness.append(fitness)
            if fitness < bbo.personal_best_fitness[i]:
                bbo.personal_best_fitness[i] = fitness
                bbo.personal_best[i] = copy.deepcopy(ind)
            if fitness < gbf:
                gbf = fitness
                gb = copy.deepcopy(ind)

        rank_order = np.argsort(bbo.fitness)
        elite_habitats = [copy.deepcopy(bbo.population[i]) for i in rank_order[:2]]
        elite_fits = [bbo.fitness[i] for i in rank_order[:2]]

        emig, immig = bbo.calculate_migration_rates()
        new_pop = copy.deepcopy(bbo.population)
        for i in range(POP_SIZE):
            if np.random.random() < emig[i]:
                mu = np.array(immig)
                s = mu.sum()
                if s > 0:
                    j = int(np.random.choice(POP_SIZE, p=mu / s))
                    if bbo.population[j]:
                        siv = np.random.randint(len(new_pop[i]))
                        if siv < len(bbo.population[j]):
                            new_pop[i][siv] = bbo.population[j][siv]
            # else: no PSO fallback — pure BBO
        bbo.population = new_pop
        bbo.fitness = bbo.fitness  # unchanged; mutation uses pre-migration order

        # Restore elites
        post_rank = np.argsort(bbo.fitness)
        for slot, (hab, fit) in enumerate(zip(elite_habitats, elite_fits)):
            bbo.population[post_rank[-(slot + 1)]] = hab
            bbo.fitness[post_rank[-(slot + 1)]] = fit

        bbo.mutation()

    best_path = bbo.decode_path(gb, connections, start_node, end_node) if gb else []
    success = bool(best_path) and (gbf < 1000)
    return (gbf if success else float("inf")), success


def run_pso_only(planner: RobotPathPlanner) -> Tuple[float, bool]:
    """Pure PSO: every habitat always uses PSO update (no migration)."""
    nodes = planner.avbn.nodes
    connections = planner.avbn.node_connections
    dist_matrix = planner.avbn.distance_matrix
    if not nodes:
        return float("inf"), False

    start_node = planner._find_nearest_node(START, nodes)
    end_node = planner._find_nearest_node(GOAL, nodes)

    pso = BPSO(pop_size=POP_SIZE, max_gen=MAX_GEN)
    pso.initialize_population(len(nodes))

    gbf = float("inf")
    gb = None

    for generation in range(MAX_GEN):
        pso._w = pso.w_max - (pso.w_max - pso.w_min) * generation / max(MAX_GEN - 1, 1)

        pso.fitness = []
        for i, ind in enumerate(pso.population):
            path = pso.decode_path(ind, connections, start_node, end_node)
            fitness = pso.calculate_fitness(path, dist_matrix, end_node)
            pso.fitness.append(fitness)
            if fitness < pso.personal_best_fitness[i]:
                pso.personal_best_fitness[i] = fitness
                pso.personal_best[i] = copy.deepcopy(ind)
            if fitness < gbf:
                gbf = fitness
                gb = copy.deepcopy(ind)

        if gb is None:
            continue
        pso.global_best = gb
        pso.global_best_fitness = gbf

        # All particles update via PSO only (no migration)
        for i in range(POP_SIZE):
            pso.update_velocity_position(i)

    best_path = pso.decode_path(gb, connections, start_node, end_node) if gb else []
    success = bool(best_path) and (gbf < 1000)
    return (gbf if success else float("inf")), success


def run_ga(planner: RobotPathPlanner) -> Tuple[float, bool]:
    """Simple GA: tournament selection + one-point crossover + bit mutation.

    Uses the same encoding/decoding/fitness as BPSO for a fair comparison.
    Paper GA parameters: crossover rate=0.9, mutation rate=0.01.
    """
    nodes = planner.avbn.nodes
    connections = planner.avbn.node_connections
    dist_matrix = planner.avbn.distance_matrix
    if not nodes:
        return float("inf"), False

    start_node = planner._find_nearest_node(START, nodes)
    end_node = planner._find_nearest_node(GOAL, nodes)

    n = len(nodes)
    P = POP_SIZE
    pc = 0.9  # crossover rate
    pm = 0.01  # mutation rate

    # Initialise
    pop = [np.random.permutation(n).tolist() for _ in range(P)]
    gbf = float("inf")
    gb = None

    _bpso = BPSO()  # used only for decode / fitness

    def evaluate(individual):
        path = _bpso.decode_path(individual, connections, start_node, end_node)
        return _bpso.calculate_fitness(path, dist_matrix, end_node)

    fits = [evaluate(ind) for ind in pop]

    for _ in range(MAX_GEN):
        # Update global best
        for i, f in enumerate(fits):
            if f < gbf:
                gbf = f
                gb = copy.deepcopy(pop[i])

        new_pop = []
        new_fits = []

        for _ in range(P // 2):
            # Tournament selection (size 3) for two parents
            def tournament():
                cands = np.random.choice(P, 3, replace=False)
                return min(cands, key=lambda c: fits[c])

            p1, p2 = tournament(), tournament()

            # One-point crossover
            if np.random.random() < pc:
                pt = np.random.randint(1, n)
                c1 = pop[p1][:pt] + pop[p2][pt:]
                c2 = pop[p2][:pt] + pop[p1][pt:]
            else:
                c1, c2 = list(pop[p1]), list(pop[p2])

            # Mutation
            for child in (c1, c2):
                for j in range(n):
                    if np.random.random() < pm:
                        child[j] = np.random.randint(0, n)

            new_pop += [c1, c2]
            new_fits += [evaluate(c1), evaluate(c2)]

        # Elitism: keep best of old generation
        best_old = int(np.argmin(fits))
        worst_new = int(np.argmax(new_fits))
        new_pop[worst_new] = copy.deepcopy(pop[best_old])
        new_fits[worst_new] = fits[best_old]

        pop = new_pop
        fits = new_fits

    success = (gb is not None) and (gbf < 1000)
    return (gbf if success else float("inf")), success


# ---------------------------------------------------------------------------
# Multi-trial runner
# ---------------------------------------------------------------------------


def run_trials(obstacles, n_trials=N_TRIALS, verbose=True):
    """Run N_TRIALS for every algorithm; return per-algorithm result lists."""
    results = {algo: [] for algo in ("GA", "PSO", "BBO", "BPSO")}

    for t in range(n_trials):
        if verbose and (t % 10 == 0):
            print(f"    Trial {t+1}/{n_trials}")

        # Build a fresh planner (and thus AVBN) for every trial.
        # All algorithms share the same network topology per trial.
        planner = _build_planner(obstacles)

        results["GA"].append(run_ga(planner))
        results["PSO"].append(run_pso_only(planner))
        results["BBO"].append(run_bbo_only(planner))
        results["BPSO"].append(run_bpso(planner))

    return results


# ---------------------------------------------------------------------------
# Plot 0 — Convergence curves  (existing, enhanced)
# ---------------------------------------------------------------------------


def plot_convergence(all_env_histories, env_names):
    """Best / mean fitness vs generation for each environment."""
    n = len(env_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, name, hist in zip(axes, env_names, all_env_histories):
        gens = range(len(hist["best_fitness"]))
        ax.plot(
            gens,
            hist["best_fitness"],
            color=ALGO_COLOURS["BPSO"],
            linewidth=2,
            label="Best fitness",
        )
        ax.plot(
            gens,
            hist["mean_fitness"],
            color=ALGO_COLOURS["BBO"],
            linewidth=1.5,
            alpha=0.7,
            linestyle="--",
            label="Mean fitness",
        )
        ax.set_title(f"{name}\nFinal cost: {hist['best_fitness'][-1]:.1f}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Fitness (path cost)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Convergence curves (BPSO) — Figure 6 equivalent", fontsize=13)
    plt.tight_layout()
    _save(fig, "plot0_convergence.png")


# ---------------------------------------------------------------------------
# Plot 1 — Success-rate comparison  (Table 2)
# ---------------------------------------------------------------------------


def plot_success_rates(all_env_results, env_names):
    """Grouped bar chart: success rate per algorithm per environment."""
    algos = ["GA", "PSO", "BBO", "BPSO"]
    n_env = len(env_names)

    # success rate = fraction of trials with cost < penalty threshold
    rates = {
        algo: [
            np.mean([s for _, s in all_env_results[env_idx][algo]])
            for env_idx in range(n_env)
        ]
        for algo in algos
    }

    x = np.arange(n_env)
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]

    fig, ax = plt.subplots(figsize=(10, 6))

    for algo, offset in zip(algos, offsets):
        bars = ax.bar(
            x + offset * width,
            rates[algo],
            width,
            label=algo,
            color=ALGO_COLOURS[algo],
            edgecolor="white",
            linewidth=0.6,
        )
        # Annotate each bar with its value
        for bar, val in zip(bars, rates[algo]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
            )

    # Reference lines from paper Table 2 — only drawn when all 3 envs present
    paper_rates = {
        "GA": [0.6, 0.8, 0.9],
        "PSO": [0.0, 0.3, 0.8],
        "BBO": [0.8, 0.9, 1.0],
        "BPSO": [1.0, 1.0, 1.0],
    }
    if n_env == 3:
        for algo, offset in zip(algos, offsets):
            ax.scatter(
                x + offset * width,
                paper_rates[algo],
                marker="_",
                s=200,
                color="black",
                linewidths=2,
                zorder=5,
                label=f"{algo} (paper)" if algo == "BPSO" else None,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(env_names)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Success rate")
    ax.set_title(
        "Plot 1 — Success-rate comparison (Table 2)\n"
        "Bars = simulated results  |  horizontal ticks = paper values",
        fontsize=11,
    )
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    _save(fig, "plot1_success_rates.png")


# ---------------------------------------------------------------------------
# Plot 2 — Box-and-whisker cost distribution  (Table 1)
# ---------------------------------------------------------------------------


def plot_cost_distribution(all_env_results, env_names):
    """One subplot per environment; one box per algorithm.

    Overlays the paper's Best / Worst / Mean values as scatter points
    so the reader can directly compare simulated spread with Table 1.
    """
    algos = ["GA", "PSO", "BBO", "BPSO"]
    n_env = len(env_names)

    # Paper Table 1 values  [Mean, Best, Worst]
    paper_table1 = {
        "Problem 1": {
            "GA": (177.2, 144, 271),
            "PSO": (164.5, 146, 172),
            "BBO": (160.2, 146, 198),
            "BPSO": (144.0, 144, 144),
        },
        "Problem 2": {
            "GA": (157.4, 153, 163),
            "PSO": (153.6, 153, 156),
            "BBO": (153.8, 153, 155),
            "BPSO": (153.0, 153, 153),
        },
        "Problem 3": {
            "GA": (122.0, 122, 155),
            "PSO": (126.7, 122, 135),
            "BBO": (122.0, 122, 122),
            "BPSO": (122.0, 122, 122),
        },
    }
    env_keys = list(paper_table1.keys())

    fig, axes = plt.subplots(1, n_env, figsize=(5 * n_env, 6), sharey=False)
    if n_env == 1:
        axes = [axes]

    for ax, env_name, env_idx in zip(axes, env_names, range(n_env)):
        # Collect cost values (successful trials only; cap failures at 500
        # so they still appear as outliers rather than dominating the axis)
        data = []
        positions = []
        for k, algo in enumerate(algos):
            costs = [min(c, 500.0) for c, _ in all_env_results[env_idx][algo]]
            data.append(costs)
            positions.append(k + 1)

        bp = ax.boxplot(
            data,
            positions=positions,
            patch_artist=True,
            widths=0.55,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker="o", markersize=4, alpha=0.5),
        )
        for patch, algo in zip(bp["boxes"], algos):
            patch.set_facecolor(ALGO_COLOURS[algo])
            patch.set_alpha(0.75)

        # Overlay paper Table 1 markers if available
        key = env_keys[env_idx] if env_idx < len(env_keys) else None
        if key and key in paper_table1:
            for k, algo in enumerate(algos):
                mean_p, best_p, worst_p = paper_table1[key][algo]
                pos = k + 1
                ax.scatter([pos], [mean_p], marker="D", s=60, color="black", zorder=6)
                ax.scatter([pos], [best_p], marker="^", s=60, color="black", zorder=6)
                ax.scatter([pos], [worst_p], marker="v", s=60, color="black", zorder=6)

        ax.set_xticks(positions)
        ax.set_xticklabels(algos)
        ax.set_title(f"{env_name}", fontsize=10)
        ax.set_ylabel("Path cost")
        ax.grid(axis="y", alpha=0.3)

    # Shared legend for paper markers
    handles = [
        mpatches.Patch(color="none", label="Paper values (black markers):"),
        plt.Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor="black",
            markersize=7,
            label="Mean",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            markerfacecolor="black",
            markersize=7,
            label="Best",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="v",
            color="w",
            markerfacecolor="black",
            markersize=7,
            label="Worst",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(
        "Plot 2 — Cost distribution over trials (Table 1 equivalent)\n"
        "Boxes = simulation  |  ▢▲▼ = paper Table 1 values",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    _save(fig, "plot2_cost_distribution.png")


# ---------------------------------------------------------------------------
# Plot 3 — Population diversity
# ---------------------------------------------------------------------------


def plot_diversity(all_env_histories, env_names):
    """Average pairwise Hamming distance between habitats vs generation."""
    n = len(env_names)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, name, hist in zip(axes, env_names, all_env_histories):
        div = hist.get("diversity", [])
        if not div:
            ax.text(0.5, 0.5, "No diversity data", transform=ax.transAxes, ha="center")
            continue

        gens = np.arange(len(div))
        ax.plot(gens, div, color=ALGO_COLOURS["BPSO"], linewidth=2)
        ax.fill_between(gens, div, alpha=0.15, color=ALGO_COLOURS["BPSO"])

        # Mark the generation where diversity is lowest (most converged)
        min_gen = int(np.argmin(div))
        ax.axvline(
            min_gen,
            color="red",
            linestyle="--",
            linewidth=1,
            label=f"Min diversity gen {min_gen}",
        )

        ax.set_title(name)
        ax.set_xlabel("Generation")
        ax.set_ylabel("Avg pairwise Hamming distance\n(normalised by genome length)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Plot 3 — Population diversity (genotypic)\n"
        "Measures whether PSO position updates prevent premature convergence",
        fontsize=11,
    )
    plt.tight_layout()
    _save(fig, "plot3_diversity.png")


# ---------------------------------------------------------------------------
# Plot 4 — Node-visit-frequency heatmap
# ---------------------------------------------------------------------------


def plot_heatmap(
    planner: RobotPathPlanner, visit_counts: np.ndarray, env_name: str, start, goal
):
    """Overlay a node-visit frequency heatmap on the AVBN environment map."""
    nodes = planner.avbn.nodes
    if not nodes or visit_counts is None:
        return

    fig, ax = plt.subplots(figsize=(10, 9))

    # Background: Voronoi-labelled obstacle map
    ax.imshow(planner.avbn.obstacle_map.T, cmap="gray_r", origin="lower", alpha=0.55)

    # Build a 2-D density grid by splatting Gaussian blobs at each node
    H, W = planner.avbn.grid_size
    heat = np.zeros((H, W), dtype=float)
    sigma = 12  # blob radius in grid cells
    max_visits = max(visit_counts.max(), 1)

    for idx, (r, c) in enumerate(nodes):
        weight = visit_counts[idx] / max_visits
        if weight < 1e-6:
            continue
        # Axis-aligned Gaussian splat
        rr = np.arange(max(0, r - 3 * sigma), min(H, r + 3 * sigma + 1))
        cc = np.arange(max(0, c - 3 * sigma), min(W, c + 3 * sigma + 1))
        RR, CC = np.meshgrid(rr, cc, indexing="ij")
        gauss = weight * np.exp(-((RR - r) ** 2 + (CC - c) ** 2) / (2 * sigma**2))
        heat[rr[0] : rr[-1] + 1, cc[0] : cc[-1] + 1] += gauss

    # Mask zero cells so they stay transparent
    heat_masked = np.ma.masked_where(heat < 1e-4, heat)
    im = ax.imshow(
        heat_masked.T,
        cmap="hot",
        origin="lower",
        alpha=0.75,
        vmin=0,
        vmax=heat.max(),
    )
    plt.colorbar(im, ax=ax, label="Visit density (normalised)", shrink=0.8)

    # Draw the AVBN network edges
    for i, (r1, c1) in enumerate(nodes):
        for j in planner.avbn.node_connections.get(i, []):
            r2, c2 = nodes[j]
            ax.plot([r1, r2], [c1, c2], color="cyan", linewidth=0.8, alpha=0.5)

    # Node scatter sized by visit count
    node_arr = np.array(nodes)
    node_size = 40 + 200 * (visit_counts / max_visits)
    ax.scatter(
        node_arr[:, 0],
        node_arr[:, 1],
        s=node_size,
        c=visit_counts,
        cmap="hot",
        edgecolors="white",
        linewidths=0.8,
        zorder=5,
    )

    # Annotate visit count on each node
    for idx, (r, c) in enumerate(nodes):
        ax.text(
            r + 4, c + 4, str(visit_counts[idx]), fontsize=7, color="white", zorder=6
        )

    # Start / goal
    ax.scatter(
        *start,
        marker="s",
        s=200,
        c="lime",
        edgecolors="black",
        linewidths=2,
        zorder=7,
        label="Start",
    )
    ax.scatter(
        *goal,
        marker="s",
        s=200,
        c="magenta",
        edgecolors="black",
        linewidths=2,
        zorder=7,
        label="Goal",
    )

    ax.legend(fontsize=9)
    ax.set_title(
        f"Plot 4 — AVBN node-visit frequency heatmap\n{env_name}\n"
        "Warmer = visited more often by BPSO population",
        fontsize=10,
    )
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    _save(fig, f"plot4_heatmap_{env_name.replace(' ', '_')}.png")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def _save(fig, filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("  Robot Path Planning — BPSO + AVBN  (Mo & Xu, 2015)")
    print("  Diagnostic plots: convergence, success rates, distribution,")
    print("  diversity, heatmap")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    env1, env2, env3 = create_test_environments()
    env_specs = [
        ("Environment 1", env1),
        ("Environment 2", env2),
        ("Environment 3", env3),
    ]
    env_names = [s[0] for s in env_specs]

    # ------------------------------------------------------------------
    # Phase 1 — single BPSO run per environment for path + history plots
    # ------------------------------------------------------------------
    single_run_histories = []
    single_run_planners = []

    for env_name, obstacles in env_specs:
        print(f"\n{'─' * 70}")
        print(f"  Single BPSO run — {env_name}")
        print(f"{'─' * 70}")

        planner = _build_planner(obstacles)
        single_run_planners.append(planner)

        path, cost, history = planner.plan_path(
            START, GOAL, pop_size=POP_SIZE, max_gen=MAX_GEN
        )
        single_run_histories.append(history)

        if path is not None:
            print(f"  Path found: {len(path)} nodes | Cost = {cost:.2f}")
            planner.visualize(
                path,
                START,
                GOAL,
                filename=os.path.join(
                    OUTPUT_DIR, f"{env_name.replace(' ', '_')}_path.png"
                ),
            )
        else:
            print("  No path found in single run.")

    # Plot 0 — convergence
    plot_convergence(single_run_histories, env_names)

    # Plot 3 — diversity (uses single-run history)
    plot_diversity(single_run_histories, env_names)

    # Plot 4 — heatmap (uses node_visit_counts from single run)
    for planner, hist, (env_name, _) in zip(
        single_run_planners, single_run_histories, env_specs
    ):
        plot_heatmap(
            planner,
            hist.get("node_visit_counts"),
            env_name,
            START,
            GOAL,
        )

    # ------------------------------------------------------------------
    # Phase 2 — multi-trial runs for statistical plots (Table 1 & 2)
    # ------------------------------------------------------------------
    print(f"\n{'─' * 70}")
    print(f"  Multi-trial phase: {N_TRIALS} trials × 4 algorithms × 3 environments")
    print(f"{'─' * 70}")

    all_env_results = []  # list[env_idx] → dict[algo] → list[(cost, success)]

    for env_name, obstacles in env_specs:
        print(f"\n  {env_name}")
        results = run_trials(obstacles, n_trials=N_TRIALS, verbose=True)
        all_env_results.append(results)

        # Print summary table
        print(
            f"\n  {'Algorithm':<8} {'Mean':>8} {'Std':>8} "
            f"{'Best':>8} {'Worst':>8} {'Success':>8}"
        )
        print("  " + "-" * 52)
        for algo in ("GA", "PSO", "BBO", "BPSO"):
            costs = [c for c, _ in results[algo]]
            valid = [c for c in costs if c < 1000]
            successes = [s for _, s in results[algo]]
            if valid:
                print(
                    f"  {algo:<8} {np.mean(valid):>8.1f} {np.std(valid):>8.2f}"
                    f" {min(valid):>8.1f} {max(valid):>8.1f}"
                    f" {np.mean(successes):>8.2f}"
                )
            else:
                print(
                    f"  {algo:<8} {'N/A':>8} {'N/A':>8} {'N/A':>8}"
                    f" {'N/A':>8} {np.mean(successes):>8.2f}"
                )

    # Plot 1 — success rates
    plot_success_rates(all_env_results, env_names)

    # Plot 2 — box-and-whisker
    plot_cost_distribution(all_env_results, env_names)

    print(f"\n{'=' * 70}")
    print("  All experiments complete.  Results in ./{OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
