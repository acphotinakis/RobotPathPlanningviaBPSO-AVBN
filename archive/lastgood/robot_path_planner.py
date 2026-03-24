import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import copy
import os

from avbn import AVBN
from bpso import BPSO


class RobotPathPlanner:
    """Complete robot path planning system using BPSO and AVBN.

    Combines the environment-modelling step (AVBN) with the optimisation
    step (BPSO) as described in §3 of Mo & Xu (2015).
    """

    def __init__(self, grid_size: Tuple[int, int] = (400, 400)):
        self.avbn = AVBN(grid_size)
        self.bpso: BPSO = None

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def setup_environment(self, obstacles: List[Tuple[int, int, int, int]]) -> None:
        """Add obstacles, enlarge them (Voronoi partition), build network."""
        self.avbn.add_obstacles(obstacles)
        self.avbn.enlarge_obstacles()
        self.avbn.build_network()

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        pop_size: int = 100,
        max_gen: int = 100,
    ):
        """Find the optimal collision-free path from *start* to *goal*.

        Returns
        -------
        path_coords : list of (row, col) waypoints along the best path
        cost        : total grid-distance cost of the best path
        history     : dict with 'best_fitness' and 'mean_fitness' lists
        """
        nodes = self.avbn.nodes
        connections = self.avbn.node_connections
        dist_matrix = self.avbn.distance_matrix

        if not nodes:
            print("No nodes found in AVBN network — cannot plan path.")
            return None, None, None

        start_node = self._find_nearest_node(start, nodes)
        end_node = self._find_nearest_node(goal, nodes)

        print(f"  Start node: {start_node}  ({nodes[start_node]})")
        print(f"  End node  : {end_node}  ({nodes[end_node]})")
        print(f"  Total nodes in network: {len(nodes)}")

        self.bpso = BPSO(pop_size=pop_size, max_gen=max_gen)

        path_indices, cost, history = self.bpso.optimize(
            nodes, connections, dist_matrix, start_node, end_node
        )

        # Convert node indices to grid coordinates
        path_coords = [nodes[i] for i in path_indices if i < len(nodes)]
        return path_coords, cost, history

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_nearest_node(
        self, point: Tuple[int, int], nodes: List[Tuple[int, int]]
    ) -> int:
        """Return the index of the node closest (Euclidean) to *point*."""
        if not nodes:
            return 0
        dists = [(point[0] - n[0]) ** 2 + (point[1] - n[1]) ** 2 for n in nodes]
        return int(np.argmin(dists))

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(
        self,
        path_indices: List[int] = None,
        start: Tuple[int, int] = None,
        goal: Tuple[int, int] = None,
        filename: str = None,
    ) -> None:
        """Render the Voronoi network, the optimal path, and save to file."""
        fig, ax = plt.subplots(figsize=(12, 10))

        # Obstacle / Voronoi map
        ax.imshow(self.avbn.obstacle_map.T, cmap="gray", origin="lower")

        # Voronoi boundary pixels
        if self.avbn.voronoi_boundaries:
            bnd = np.array(self.avbn.voronoi_boundaries)
            ax.scatter(
                bnd[:, 0],
                bnd[:, 1],
                c="blue",
                s=1,
                alpha=0.3,
                label="Voronoi boundaries",
            )

        # Network nodes
        if self.avbn.nodes:
            nd = np.array(self.avbn.nodes)
            ax.scatter(
                nd[:, 0], nd[:, 1], c="red", s=50, marker="o", label="Network nodes"
            )

        # Optimal path: stitch pre-computed boundary segments
        if path_indices and hasattr(self.avbn, "path_matrix"):
            full_path: List[Tuple[int, int]] = []
            for k in range(len(path_indices) - 1):
                n1, n2 = path_indices[k], path_indices[k + 1]
                seg = self.avbn.path_matrix.get(n1, {}).get(n2)
                if seg:
                    full_path.extend(seg)

            if full_path:
                pth = np.array(full_path)
                ax.plot(pth[:, 0], pth[:, 1], "g-", linewidth=3, label="Optimal path")

                macro = np.array(
                    [
                        self.avbn.nodes[idx]
                        for idx in path_indices
                        if idx < len(self.avbn.nodes)
                    ]
                )
                if len(macro):
                    ax.scatter(
                        macro[:, 0], macro[:, 1], c="green", s=100, marker="*", zorder=5
                    )

        # Start / goal markers
        if start:
            ax.scatter(
                *start,
                c="yellow",
                s=200,
                marker="s",
                edgecolors="black",
                linewidths=2,
                label="Start",
                zorder=6,
            )
        if goal:
            ax.scatter(
                *goal,
                c="purple",
                s=200,
                marker="s",
                edgecolors="black",
                linewidths=2,
                label="Goal",
                zorder=6,
            )

        ax.legend()
        ax.set_title("Robot Path Planning — BPSO + AVBN")
        ax.set_xlabel("X coordinate")
        ax.set_ylabel("Y coordinate")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        if filename:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            plt.savefig(filename, dpi=150)
            print(f"  Plot saved --> {filename}")

        plt.show()
        plt.close(fig)
