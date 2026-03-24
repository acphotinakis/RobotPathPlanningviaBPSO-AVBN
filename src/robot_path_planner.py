# =============================================================================
# robot_path_planner.py — Top-level orchestration: AVBN + BPSO
# =============================================================================
#
# PAPER REFERENCE
# ---------------
# Mo, H. & Xu, L. (2015). "Research of biogeography particle swarm
# optimization for robot path planning." Neurocomputing, 148, 91-99.
# https://doi.org/10.1016/j.neucom.2012.07.060
#
# This file implements the top-level path-planning procedure from Section 3
# of the paper, which combines the AVBN environment model (Section 3.1) with
# the BPSO optimiser (Section 2) to produce the AVBN-based BPSO path-planning
# algorithm described in Section 3.2.
#
# HOW TO READ THE ANNOTATIONS
# ----------------------------
# Each annotation block immediately precedes the code it describes:
#
#   [Paper §X.Y]   — location in the paper
#   Quote: "..."   — verbatim sentence(s) from the paper
#   Mapping:       — code ↔ paper concept correspondence
#
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import copy
import os

from avbn import AVBN
from bpso import BPSO


class RobotPathPlanner:
    # -------------------------------------------------------------------------
    # [Paper §3 — The path planning based on AVBN]
    # Quote: "In order to calculate the trajectory in the global map, this
    #         paper presents a new RPP method based on the combination of BPSO
    #         and AVBN method. The biogeography based particle swarm
    #         optimization algorithm is adopted to search the possible paths
    #         and the best one is obtained."
    # Mapping:
    #   self.avbn   — the AVBN environment model (Section 3.1); responsible
    #                 for building the Voronoi network that BPSO searches over.
    #   self.bpso   — the BPSO optimiser (Section 2); operates on the network
    #                 produced by avbn to find the lowest-cost path.
    # -------------------------------------------------------------------------

    """Complete robot path planning system using BPSO and AVBN.

    Combines the environment-modelling step (AVBN) with the optimisation
    step (BPSO) as described in §3 of Mo & Xu (2015).
    """

    def __init__(self, grid_size: Tuple[int, int] = (400, 400)):
        self.avbn = AVBN(grid_size)
        self.bpso: BPSO = None

    # =========================================================================
    # Section 3.1 — Environment setup pipeline
    # =========================================================================

    def setup_environment(self, obstacles: List[Tuple[int, int, int, int]]) -> None:
        # ---------------------------------------------------------------------
        # [Paper §3.1.2 — Construction of AVBN model]
        # Quote: "The obstacles detected by sensor are discrete grid data.
        #         At first, the obstacle grid is enlarged to its neighbour
        #         grids, so some small obstacle clusters are fused to an
        #         integrated obstacle block."
        # Quote: "The AVBN method can establish the feasible paths network of
        #         mobile robot. Thus, it can transfer the path planning problem
        #         into the shortest path between two route nodes in network."
        #
        # Mapping: This method is the three-stage AVBN construction pipeline:
        #   add_obstacles()     — label each obstacle F_i in the integer matrix
        #                         (§3.1 / §3.1.2 first paragraph)
        #   enlarge_obstacles() — raster-based uniform enlargement (§3.1.2.1)
        #                         until all free cells are absorbed into the
        #                         Voronoi partition
        #   build_network()     — detect boundary cells (§3.1.2.2 bullet 1),
        #                         find intersection nodes (§3.1.2.2 bullet 2),
        #                         and build the correlated nodes table (§3.1.3)
        # ---------------------------------------------------------------------

        """Add obstacles, enlarge them (Voronoi partition), build network."""
        self.avbn.add_obstacles(obstacles)
        # Save a copy BEFORE enlargement
        before = self.avbn.obstacle_map.copy()
        self.avbn.enlarge_obstacles()
        # After enlargement
        after = self.avbn.obstacle_map
        self.avbn.plot_obstacles_before_after(before, after)
        self.avbn.plot_obstacles()
        self.avbn.build_network()
        self.avbn.plot_distance_matrix()
        self.avbn.plot_voronoi_boundaries()
        self.avbn.plot_network()
        self.avbn.plot_correlated_nodes_table()
        import sys

        sys.exit(0)

    # =========================================================================
    # Section 3.2 — AVBN-based BPSO path-planning algorithm
    # =========================================================================

    def plan_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        pop_size: int = 100,
        max_gen: int = 100,
    ):
        # ---------------------------------------------------------------------
        # [Paper §3.1.3 — Global path planning / §3.2 — AVBN based BPSO]
        # Quote: "A mobile robot should search the shortcut to the network when
        #         it plans to move from a start point to a destination point.
        #         From the start point S and the destination point Q, search the
        #         shortest distance point in AVBN individually, and the
        #         intersection is named start node (node 1 in Fig. 2) and end
        #         node (node 15 in Fig. 2). For example, from the start point,
        #         search the intersecting points on the network in every
        #         direction and chose the point as the start node which has the
        #         shortest distance to the start point."
        # Quote (§4.2 parameters): "the number of initial population is 100.
        #         The maximal iteration generation number is 100."
        #
        # Mapping:
        #   start, goal       — S (start point) and Q (destination point) from
        #                       the paper's §3.1.3 description
        #   start_node        — the network node closest to S; "node 1" in
        #                       the Fig. 2 example
        #   end_node          — the network node closest to Q; "node 15" in
        #                       the Fig. 2 example
        #   _find_nearest_node() — implements "search the shortest distance
        #                       point in AVBN individually" using Euclidean
        #                       distance to all node coordinates
        #   pop_size = 100    — P = 100 (population size, §4.2)
        #   max_gen  = 100    — G = 100 (max generation number, §4.2)
        #   bpso.optimize()   — the full BPSO algorithm (§2.3) operating on
        #                       the AVBN network; returns path_indices (a
        #                       sequence of node indices S_i), the best cost
        #                       C(G_i), and the per-generation history.
        #   path_coords       — the sequence of (row, col) grid coordinates
        #                       corresponding to S_i; the paper refers to
        #                       these as "route nodes" of the optimal path.
        # ---------------------------------------------------------------------
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

    # =========================================================================
    # Helper: nearest-node lookup (§3.1.3)
    # =========================================================================

    def _find_nearest_node(
        self, point: Tuple[int, int], nodes: List[Tuple[int, int]]
    ) -> int:
        # ---------------------------------------------------------------------
        # [Paper §3.1.3 — Global path planning]
        # Quote: "search the intersecting points on the network in every
        #         direction and chose the point as the start node which has
        #         the shortest distance to the start point. The end node is
        #         searched out in the same way."
        # Mapping:
        #   point     — either S (start) or Q (goal)
        #   nodes     — the list of all AVBN intersection nodes
        #   dists     — squared Euclidean distances from *point* to every
        #               node; we use squared distance (no sqrt) because we
        #               only need the argmin, not the actual distance value.
        #   argmin    — index of the closest node: the "start node" or
        #               "end node" as defined in §3.1.3.
        # ---------------------------------------------------------------------
        """Return the index of the node closest (Euclidean) to *point*."""
        if not nodes:
            return 0
        dists = [(point[0] - n[0]) ** 2 + (point[1] - n[1]) ** 2 for n in nodes]
        return int(np.argmin(dists))

    # =========================================================================
    # Visualisation — renders Fig. 5 equivalent (best path in environment)
    # =========================================================================

    def visualize(
        self,
        path_indices: List[int] = None,
        start: Tuple[int, int] = None,
        goal: Tuple[int, int] = None,
        filename: str = None,
    ) -> None:
        # ---------------------------------------------------------------------
        # [Paper §4 — Simulation results / Fig. 5]
        # Quote: "The best paths of three grid environments by BPSO are shown
        #         in Fig. 5(a)–(c), respectively."
        # Quote: "The path is a curve obtained by enlarging obstacles, so the
        #         best path in AVBN is not certain in reality." (§3.2)
        #
        # Mapping:
        #   obstacle_map.T       — the Voronoi-partitioned integer grid shown
        #                          in greyscale; corresponds to Fig. 1(d) /
        #                          the coloured backgrounds in Fig. 5.
        #   voronoi_boundaries   — the B_{i,j} boundary pixel sets from
        #                          Definition 3; shown as blue scatter points,
        #                          forming the "non-smooth routes" of the AVBN.
        #   nodes                — the intersection nodes numbered in Fig. 2;
        #                          shown as red circles.
        #   path_matrix[n1][n2]  — the pixel-level coordinates of each branch
        #                          in the best decoded path S_i; stitched
        #                          together to draw the continuous green path
        #                          matching Fig. 5.
        #   start, goal markers  — S (start point) and Q (destination point)
        #                          labelled in Fig. 5 of the paper.
        # ---------------------------------------------------------------------
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
