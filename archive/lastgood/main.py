import matplotlib.pyplot as plt
import os

from robot_path_planner import RobotPathPlanner


# ---------------------------------------------------------------------------
# Test environments (Figure 4 from the paper)
# ---------------------------------------------------------------------------


def create_test_environments():
    """Return the three obstacle configurations from Figure 4."""

    # Environment 1 — random scattered obstacles
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

    # Environment 2 — corridor-like obstacles
    env2 = [
        (50, 50, 20, 80),
        (150, 30, 80, 20),
        (250, 50, 20, 80),
        (100, 150, 20, 80),
        (200, 180, 80, 20),
        (150, 250, 20, 50),
    ]

    # Environment 3 — complex irregular obstacles
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
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 65)
    print("  Robot Path Planning — BPSO + AVBN  (Mo & Xu, 2015)")
    print("=" * 65)

    os.makedirs("output", exist_ok=True)

    env1, env2, env3 = create_test_environments()
    environments = [
        ("Environment 1", env1, (10, 10), (350, 350)),
        ("Environment 2", env2, (10, 10), (350, 350)),
        ("Environment 3", env3, (10, 10), (350, 350)),
    ]

    results = []

    for env_name, obstacles, start, goal in environments:
        print(f"\n{'=' * 65}")
        print(f"  {env_name}")
        print(f"{'=' * 65}")

        planner = RobotPathPlanner(grid_size=(400, 400))

        print("Setting up environment …")
        planner.setup_environment(obstacles)

        print(f"Planning path {start} --> {goal} …")
        path, cost, history = planner.plan_path(start, goal, pop_size=100, max_gen=100)

        if path is not None:
            print(f"\n  Path found!  Nodes = {len(path)}  |  Cost = {cost:.2f}")
            results.append(
                {
                    "environment": env_name,
                    "cost": cost,
                    "path_length": len(path),
                    "history": history,
                }
            )
            planner.visualize(
                path,
                start,
                goal,
                filename=f"output/{env_name.replace(' ', '_')}_path.png",
            )
        else:
            print("  No path found.")

    # ------------------------------------------------------------------
    # Convergence comparison plot (Figure 6 equivalent)
    # ------------------------------------------------------------------
    if results:
        fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
        if len(results) == 1:
            axes = [axes]

        for ax, result in zip(axes, results):
            ax.plot(result["history"]["best_fitness"], label="Best fitness")
            ax.plot(result["history"]["mean_fitness"], label="Mean fitness", alpha=0.6)
            ax.set_xlabel("Generation")
            ax.set_ylabel("Fitness (path cost)")
            ax.set_title(f"{result['environment']}\nFinal cost: {result['cost']:.2f}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = "output/convergence_comparison.png"
        plt.savefig(out_path, dpi=150)
        print(f"\nConvergence plot saved --> {out_path}")
        plt.show()
        plt.close(fig)

    print("\n" + "=" * 65)
    print("  All experiments complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()
