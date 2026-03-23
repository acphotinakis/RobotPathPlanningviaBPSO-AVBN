import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from typing import List, Tuple, Dict
import copy

from avbn import AVBN
from bpso import BPSO


class RobotPathPlanner:
    """Complete robot path planning system using BPSO and AVBN"""

    def __init__(self, grid_size: Tuple[int, int] = (400, 400)):
        self.avbn = AVBN(grid_size)
        self.bpso = None

    def setup_environment(
        self, obstacles: List[Tuple[int, int, int, int]], enlarge_iterations: int = 5
    ):
        """Setup the environment with obstacles"""
        self.avbn.add_obstacles(obstacles)
        # self.avbn.enlarge_obstacles(enlarge_iterations)
        self.avbn.enlarge_obstacles()
        self.avbn.build_network()

    def plan_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        pop_size: int = 100,
        max_gen: int = 100,
    ):
        """Plan optimal path from start to goal"""
        nodes, connections = self.avbn.nodes, self.avbn.node_connections

        if len(nodes) == 0:
            print("No nodes found in network!")
            return None, None, None

        # Find nearest nodes to start and goal
        start_node = self._find_nearest_node(start, nodes)
        end_node = self._find_nearest_node(goal, nodes)

        print(f"Start node: {start_node}, End node: {end_node}")
        print(f"Total nodes: {len(nodes)}")

        # Initialize BPSO
        self.bpso = BPSO(pop_size=pop_size, max_gen=max_gen)

        # Optimize path
        path, cost, history = self.bpso.optimize(
            nodes, connections, self.avbn.obstacle_map, start_node, end_node
        )

        # Convert node indices to coordinates
        path_coords = [nodes[i] for i in path if i < len(nodes)]

        return path_coords, cost, history

    def _find_nearest_node(
        self, point: Tuple[int, int], nodes: List[Tuple[int, int]]
    ) -> int:
        """Find nearest network node to a point"""
        if not nodes:
            return 0

        distances = [
            np.sqrt((point[0] - n[0]) ** 2 + (point[1] - n[1]) ** 2) for n in nodes
        ]
        return int(np.argmin(distances))

    def visualize(
        self,
        path_coords: List[Tuple[int, int]] = None,
        start: Tuple[int, int] = None,
        goal: Tuple[int, int] = None,
        filename: str = None,  # Added filename parameter
    ):
        """Visualize the environment and path and save to file"""
        plt.figure(figsize=(12, 10))

        # Show obstacle map (transposed and flipped to match paper's coordinate system)
        plt.imshow(self.avbn.obstacle_map.T, cmap="gray", origin="lower")

        # Show Voronoi boundaries
        if self.avbn.voronoi_boundaries:
            boundaries = np.array(self.avbn.voronoi_boundaries)
            plt.scatter(
                boundaries[:, 0],
                boundaries[:, 1],
                c="blue",
                s=1,
                alpha=0.3,
                label="Voronoi Boundaries",
            )

        # Show nodes
        if self.avbn.nodes:
            nodes = np.array(self.avbn.nodes)
            plt.scatter(
                nodes[:, 0], nodes[:, 1], c="red", s=50, marker="o", label="Nodes"
            )

        # Show path
        if path_coords:
            path = np.array(path_coords)
            plt.plot(path[:, 0], path[:, 1], "g-", linewidth=3, label="Optimal Path")
            plt.scatter(path[:, 0], path[:, 1], c="green", s=100, marker="*")

        # Show start and goal
        if start:
            plt.scatter(
                start[0],
                start[1],
                c="yellow",
                s=200,
                marker="s",
                edgecolors="black",
                linewidths=2,
                label="Start",
            )
        if goal:
            plt.scatter(
                goal[0],
                goal[1],
                c="purple",
                s=200,
                marker="s",
                edgecolors="black",
                linewidths=2,
                label="Goal",
            )

        plt.legend()
        plt.title("Robot Path Planning using BPSO-AVBN")
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save logic
        if filename:
            import os

            # Ensure output directory exists
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename)
            print(f"Plot saved to: {filename}")

        plt.show()
        plt.close()  # Close figure to free memory during loop

    # def visualize(
    #     self,
    #     path_coords: List[Tuple[int, int]] = None,
    #     start: Tuple[int, int] = None,
    #     goal: Tuple[int, int] = None,
    # ):
    #     """Visualize the environment and path"""
    #     plt.figure(figsize=(12, 10))

    #     # Show obstacle map
    #     plt.imshow(self.avbn.obstacle_map.T, cmap="gray", origin="lower")

    #     # Show Voronoi boundaries
    #     if self.avbn.voronoi_boundaries:
    #         boundaries = np.array(self.avbn.voronoi_boundaries)
    #         plt.scatter(
    #             boundaries[:, 0],
    #             boundaries[:, 1],
    #             c="blue",
    #             s=1,
    #             alpha=0.3,
    #             label="Voronoi Boundaries",
    #         )

    #     # Show nodes
    #     if self.avbn.nodes:
    #         nodes = np.array(self.avbn.nodes)
    #         plt.scatter(
    #             nodes[:, 0], nodes[:, 1], c="red", s=50, marker="o", label="Nodes"
    #         )

    #     # Show path
    #     if path_coords:
    #         path = np.array(path_coords)
    #         plt.plot(path[:, 0], path[:, 1], "g-", linewidth=3, label="Optimal Path")
    #         plt.scatter(path[:, 0], path[:, 1], c="green", s=100, marker="*")

    #     # Show start and goal
    #     if start:
    #         plt.scatter(
    #             start[0],
    #             start[1],
    #             c="yellow",
    #             s=200,
    #             marker="s",
    #             edgecolors="black",
    #             linewidths=2,
    #             label="Start",
    #         )
    #     if goal:
    #         plt.scatter(
    #             goal[0],
    #             goal[1],
    #             c="purple",
    #             s=200,
    #             marker="s",
    #             edgecolors="black",
    #             linewidths=2,
    #             label="Goal",
    #         )

    #     plt.legend()
    #     plt.title("Robot Path Planning using BPSO-AVBN")
    #     plt.xlabel("X coordinate")
    #     plt.ylabel("Y coordinate")
    #     plt.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.show()
