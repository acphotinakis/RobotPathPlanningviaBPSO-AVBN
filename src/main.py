import matplotlib.pyplot as plt

from robot_path_planner import RobotPathPlanner


# Example usage demonstrating the three test environments from the paper
def create_test_environments():
    """Create the three test environments from Figure 4"""

    # Environment 1: Random scattered obstacles
    env1_obstacles = [
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

    # Environment 2: Corridor-like obstacles
    env2_obstacles = [
        (50, 50, 20, 80),
        (150, 30, 80, 20),
        (250, 50, 20, 80),
        (100, 150, 20, 80),
        (200, 180, 80, 20),
        (150, 250, 20, 50),
    ]

    # Environment 3: Complex irregular obstacles
    env3_obstacles = [
        (50, 50, 30, 40),
        (50, 90, 20, 30),
        (70, 80, 30, 10),
        (150, 100, 50, 50),
        (170, 150, 40, 30),
        (190, 130, 30, 20),
        (250, 200, 40, 40),
        (100, 250, 40, 30),
    ]

    return env1_obstacles, env2_obstacles, env3_obstacles


def main():
    """Main function to run path planning experiments"""

    print("=" * 60)
    print("Robot Path Planning using BPSO and AVBN")
    print("=" * 60)

    # Create test environments
    env1, env2, env3 = create_test_environments()
    environments = [
        ("Environment 1", env1, (10, 10), (350, 350)),
        ("Environment 2", env2, (10, 10), (350, 350)),
        ("Environment 3", env3, (10, 10), (350, 350)),
    ]

    results = []

    for env_name, obstacles, start, goal in environments:
        print(f"\n{'=' * 60}")
        print(f"Testing {env_name}")
        print(f"{'=' * 60}")

        # Create planner
        planner = RobotPathPlanner(grid_size=(400, 400))

        # Setup environment
        print("Setting up environment...")
        planner.setup_environment(obstacles, enlarge_iterations=3)

        # Plan path
        print(f"Planning path from {start} to {goal}...")
        path, cost, history = planner.plan_path(start, goal, pop_size=100, max_gen=100)

        if path:
            print(f"\nPath found!")
            print(f"Path length: {len(path)} nodes")
            print(f"Total cost: {cost:.2f}")

            results.append(
                {
                    "environment": env_name,
                    "cost": cost,
                    "path_length": len(path),
                    "history": history,
                }
            )

            # Visualize
            planner.visualize(
                path, start, goal, filename="../output/robot_path_planner.png"
            )
        else:
            print("No path found!")

    # Plot convergence comparison
    if results:
        plt.figure(figsize=(14, 5))

        for idx, result in enumerate(results):
            plt.subplot(1, 3, idx + 1)
            plt.plot(result["history"]["best_fitness"], label="Best Fitness")
            plt.plot(result["history"]["mean_fitness"], label="Mean Fitness", alpha=0.6)
            plt.xlabel("Generation")
            plt.ylabel("Fitness (Path Cost)")
            plt.title(f"{result['environment']}\nFinal Cost: {result['cost']:.2f}")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        # plt.show()
        plt.savefig("../output/main_plot_convergence_comparison.png")

    print("\n" + "=" * 60)
    print("Experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
