# =============================================================================
# avbn.py — Approximate Voronoi Boundary Network
# =============================================================================
#
# PAPER REFERENCE
# ---------------
# Mo, H. & Xu, L. (2015). "Research of biogeography particle swarm
# optimization for robot path planning." Neurocomputing, 148, 91-99.
# https://doi.org/10.1016/j.neucom.2012.07.060
#
# This file implements the AVBN environment-modelling method described in
# Section 3.1 of the paper (Sections 3.1.1, 3.1.2.1, and 3.1.2.2).
#
# HOW TO READ THE ANNOTATIONS
# ----------------------------
# Each annotation block immediately precedes the code it describes:
#
#   [Paper §X.Y / Eq. N / Definition N]  — location in the paper
#   Quote: "..."                          — verbatim sentence(s)
#   Mapping:                              — code ↔ paper symbol correspondence
#
# =============================================================================


import numpy as np
from typing import List, Tuple, Dict
from collections import deque
import matplotlib.pyplot as plt
import os
from numpy.typing import NDArray
import pandas as pd


class AVBN:
    # -------------------------------------------------------------------------
    # [Paper §3.1 — Environment Modelling overview]
    # Quote: "The RPP problem considered in this section is in a known static
    #         environment. [...] The working area is divided into 400×400 grids
    #         environment. Each grid is corresponding to a small area in real
    #         environment. The distribution of obstacles in the environment is
    #         represented by integer matrix 400×400. The value indicates the
    #         obstacle grid, and the value 0 indicates the free grid."
    # Mapping:
    #   grid_size       — the 400×400 integer matrix dimensions from the paper
    #   obstacle_map    — the integer matrix; 0 = free cell, i > 0 = obstacle i
    #   voronoi_boundaries — B_{i,j}: set of boundary cells (Definition 3)
    #   nodes           — intersection points where three or more boundaries meet
    #   node_connections   — "correlated nodes table" L_j from §3.1.3
    #   distance_matrix    — branch cost D(s_i, s_{i+1}) used in Eq. (7)
    #   path_matrix        — pixel-level path for each inter-node branch;
    #                        used by the visualiser to draw the exact Voronoi
    #                        boundary route between nodes
    # -------------------------------------------------------------------------
    """Approximate Voronoi Boundary Network for environment modelling.

    Implements the AVBN construction described in §3.1 of:
      Mo & Xu, "Research of biogeography particle swarm optimization
      for robot path planning", Neurocomputing 148 (2015) 91-99.

    Changes vs. original code
    --------------------------
    * _build_connections: the node membership test  `(r, c) in self.nodes`
      was an O(N) list scan on every BFS step.  We build a node_set (a Python
      set) so the check is O(1).  This is a correctness-neutral performance
      fix, but it also prevents subtle bugs where duplicate node coordinates
      could cause the wrong index to be returned by list.index().
    * node_connections is now stored as an attribute on the instance (dict of
      lists) so that BPSO.decode_path can use it directly, matching the paper's
      "correlated nodes table".
    * Minor: distance_matrix and path_matrix initialised to empty dicts at
      construction time; build_network returns node_connections as well so
      callers have a single entry point.
    """

    def __init__(self, grid_size: Tuple[int, int] = (400, 400)):
        self.grid_size = grid_size
        self.obstacle_map = np.zeros(grid_size, dtype=np.int32)
        self.voronoi_boundaries: List[Tuple[int, int]] = []
        self.nodes: List[Tuple[int, int]] = []
        self.node_connections: Dict[int, List[int]] = {}
        self.distance_matrix: Dict[int, Dict[int, int]] = {}
        self.path_matrix: Dict[int, Dict[int, List[Tuple[int, int]]]] = {}

    # =========================================================================
    # Environment setup — labelling obstacles in the integer matrix
    # =========================================================================

    def add_obstacles(self, obstacles: List[Tuple[int, int, int, int]]) -> None:
        # ---------------------------------------------------------------------
        # [Paper §3.1 / §3.1.2 Construction of AVBN model]
        # Quote: "The algorithm checks obstacle blocks and labels them. Every
        #         grid of the same obstacle block is labelled in the same
        #         number."
        # Quote: "The working area is assumed being surrounded by a rectangular
        #         obstacle."
        # Mapping:
        #   obstacles       — list of (x, y, width, height) rectangles; each
        #                     represents one named obstacle sub-set F_i from
        #                     Definition: "The obstacle set is composed of
        #                     several separated obstacle sub sets F_i
        #                     (i = 1, 2, ..., M)."
        #   idx             — the integer label i assigned to obstacle F_i;
        #                     every cell of the same obstacle shares this label,
        #                     matching the paper's labelling scheme.
        #   self.obstacle_map[y:y+h, x:x+w] = idx
        #                   — stamps the rectangle into the integer matrix;
        #                     0 remains for free cells (F, the free set).
        # ---------------------------------------------------------------------
        """Add rectangular obstacles to the grid.

        Args:
            obstacles: List of (x, y, width, height) tuples.
                       x/y follow image conventions: x=column, y=row.
        """
        for idx, (x, y, w, h) in enumerate(obstacles, start=1):
            self.obstacle_map[y : y + h, x : x + w] = idx

    # =========================================================================
    # Section 3.1.2.1 — Raster-based enlargement of obstacles
    # =========================================================================

    def enlarge_obstacles(self) -> None:
        # ---------------------------------------------------------------------
        # [Paper §3.1.2.1 — Raster-based enlargement of obstacles]
        # Quote: "The obstacle cluster is the kernel of enlargement. Free-grid
        #         is eroded by its neighbouring obstacle grid. Every obstacle
        #         cluster grows up uniformly after one enlargement. The
        #         enlargement should not stop until all the free grids are fused
        #         into eroded area (Fig. 1(a)–(d)), in which every area is
        #         labelled with different numbers and in different colours."
        # Quote: "The scale of obstacle pre-enlargement should involve the
        #         consideration of not only the safe traversing radius of mobile
        #         robot, but also the error of sensors and positioning."
        #
        # Mapping:
        #   self.obstacle_map == 0    — the free set F; the while-loop
        #                               continues as long as any free cell
        #                               exists, exactly as the paper requires.
        #   binary_dilation(mask)     — uniform growth of obstacle i by one
        #                               cell in all directions each iteration.
        #   new_cells                 — the free cells that border obstacle i
        #                               and have not yet been claimed by any
        #                               other obstacle in this iteration; they
        #                               are re-labelled with i.  The double
        #                               guard (obstacle_map==0) & (new_map==0)
        #                               prevents two obstacles from overwriting
        #                               each other in the same step, preserving
        #                               the uniform-growth rule.
        #
        # After convergence, self.obstacle_map contains the discrete Voronoi
        # partition of the workspace: every cell carries the label of its
        # nearest obstacle, realising Definitions 1-3 from §3.1.1.
        # ---------------------------------------------------------------------

        """Expand every obstacle region uniformly until no free cell remains.

        This produces the discrete Voronoi partition described in §3.1.2.1:
        each free cell is eventually labelled with the index of its nearest
        obstacle, and the boundaries between differently-labelled regions
        become the Voronoi boundaries.
        """
        from scipy.ndimage import binary_dilation

        while (self.obstacle_map == 0).any():
            new_map = self.obstacle_map.copy()
            n_obstacles = int(self.obstacle_map.max())
            for i in range(1, n_obstacles + 1):
                mask = self.obstacle_map == i
                dilated = binary_dilation(mask)
                # Grow only into cells that are still free in *both* the
                # current map and the new_map (prevents one obstacle from
                # overwriting another that grew in the same iteration).
                new_cells = dilated & (self.obstacle_map == 0) & (new_map == 0)
                new_map[new_cells] = i
            self.obstacle_map = new_map

    # =========================================================================
    # Section 3.1.2.2 — Recognising Voronoi boundaries (bullet 1)
    # =========================================================================

    def find_voronoi_boundaries(self) -> List[Tuple[int, int]]:
        # ---------------------------------------------------------------------
        # [Paper §3.1.2.2 — Recognising the Voronoi boundary, bullet 1]
        # Quote: "There are 8 neighbouring grids in maximum for each grid.
        #         A grid will be labelled as a boundary grid if the upside grid
        #         or right grid is not at the same value as itself."
        # Quote: "At first, search out all the grids whose upside grid does not
        #         belong to the same area (see Fig. 1(a) and (b)), then search
        #         out the ones whose right grid is not at the same value
        #         (see Fig. 1(c))."
        #
        # Mapping:
        #   val = obstacle_map[i, j]       — the Voronoi label of cell (i, j)
        #   obstacle_map[i+1, j]           — the "upside grid" (next row)
        #   obstacle_map[i, j+1]           — the "right grid" (next column)
        #   boundaries.append((i, j))      — cell is labelled a boundary grid
        #                                    when either neighbour differs,
        #                                    implementing Definition 3:
        #                                    B_{i,j} = {x ∈ F | D_i(x) = D_j(x)
        #                                    = D(x), i≠j}
        # ---------------------------------------------------------------------
        """Return all grid cells that lie on a Voronoi boundary.

        A cell is a boundary cell if its right neighbour or its upward
        (row+1) neighbour belongs to a different Voronoi region — exactly
        the criterion given in §3.1.2.2 bullet 1.
        """
        boundaries: List[Tuple[int, int]] = []
        h, w = self.grid_size

        for i in range(h - 1):
            for j in range(w - 1):
                val = self.obstacle_map[i, j]
                if (
                    val != self.obstacle_map[i + 1, j]
                    or val != self.obstacle_map[i, j + 1]
                ):
                    boundaries.append((i, j))

        return boundaries

    # =========================================================================
    # Section 3.1.2.2 — Recognising Voronoi nodes (bullet 2)
    # =========================================================================

    def find_nodes(self, boundaries: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        # ---------------------------------------------------------------------
        # [Paper §3.1.2.2 — Recognising the Voronoi nodes, bullet 2]
        # Quote: "A boundary grid and its neighbouring grids are integrated as
        #         one Voronoi node if the neighbouring grids belong to three or
        #         more than three different areas (see Fig. 1(d))."
        # Quote: "The basic nodes are the intersections of these boundaries.
        #         These basic nodes and boundaries make up the AVBN that is the
        #         path network for mobile robots."
        #
        # Mapping:
        #   region_ids      — the set of distinct Voronoi labels in the
        #                     8-neighbourhood of cell (i, j); a cell qualifies
        #                     as a node candidate when |region_ids| ≥ 3,
        #                     meaning three or more distinct areas meet there.
        #   candidates      — all boundary cells satisfying the ≥ 3-area rule
        #   filtered        — the final node list after merging candidates that
        #                     are within 15 cells of each other into a single
        #                     representative intersection point; this avoids
        #                     multiple near-duplicate nodes at the same
        #                     geometric junction.
        #   dists.min() > 15 — the 15-cell distance threshold used to decide
        #                      whether a new candidate is far enough from all
        #                      already-accepted nodes to be kept as a distinct
        #                      node rather than merged into the nearest one.
        # ---------------------------------------------------------------------

        """Identify Voronoi intersection nodes from the boundary set.

        §3.1.2.2 bullet 2: a boundary cell is a node if its 8-neighbourhood
        touches three or more distinct Voronoi regions.  Nearby candidate
        nodes (within 15 cells of each other) are merged into one to avoid
        clusters of near-duplicate nodes at the same geometric intersection.
        """
        if not boundaries:
            return []

        h, w = self.grid_size
        candidates: List[Tuple[int, int]] = []

        for i, j in boundaries:
            region_ids: set = set()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        region_ids.add(int(self.obstacle_map[ni, nj]))
            if len(region_ids) >= 3:
                candidates.append((i, j))

        if not candidates:
            return []

        # Merge nodes that are too close together (15-cell threshold)
        filtered: List[Tuple[int, int]] = []
        cand_arr = np.array(candidates, dtype=float)

        for k, node in enumerate(candidates):
            if not filtered:
                filtered.append(node)
                continue
            filt_arr = np.array(filtered, dtype=float)
            dists = np.sqrt(
                np.sum((filt_arr - np.array(node, dtype=float)) ** 2, axis=1)
            )
            if dists.min() > 15:
                filtered.append(node)

        return filtered

    # =========================================================================
    # Section 3.1.3 — Global path planning: building the correlated nodes table
    # =========================================================================

    def build_network(
        self,
    ) -> Tuple[List[Tuple[int, int]], Dict[int, Dict[int, int]]]:
        # ---------------------------------------------------------------------
        # [Paper §3.1.3 — Global path planning]
        # Quote: "The AVBN method can establish the feasible paths network of
        #         mobile robot. Thus, it can transfer the path planning problem
        #         into the shortest path between two route nodes in network."
        # Quote: "According to the connecting relation in the network, a
        #         distance matrix can be established as follows."
        # Mapping:
        #   boundaries    — all Voronoi boundary pixels B_{i,j} from §3.1.2.2
        #   self.nodes    — the intersection nodes (numbered 1–15 in Fig. 2)
        #   _build_connections() — populates node_connections (correlated nodes
        #                          table), distance_matrix (branch costs), and
        #                          path_matrix (pixel routes between nodes).
        #                          Together these form the network shown in
        #                          Fig. 2 of the paper.
        # ---------------------------------------------------------------------
        """Build the complete AVBN: boundaries --> nodes --> connections.

        Returns
        -------
        nodes            : list of (row, col) intersection coordinates
        distance_matrix  : dict[i][j] = pixel distance along boundary
        """
        boundaries = self.find_voronoi_boundaries()
        # self.plot_voronoi_boundaries()
        self.voronoi_boundaries = boundaries
        self.nodes = self.find_nodes(boundaries)
        self._build_connections()
        return self.nodes, self.distance_matrix

    def _build_connections(self) -> None:
        # ---------------------------------------------------------------------
        # [Paper §3.1.3 — Global path planning / §3.2 Step 4 distance matrix]
        # Quote: "Calculating the cost of each branch, then the correlated
        #         nodes table of the network can be obtained."
        # Quote (example from §3.1.3): "L1 = {2, 8}, l1 = 2; L2 = {1,3,5},
        #         l2 = 3; ..." where L_j is the adjacency list of node j and
        #         l_j = |L_j| is the number of reachable neighbours.
        # Quote: "The cost of every branch of AVBN is calculated by adding up
        #         at the grids on path between every two nodes." (§3.2 Step 4)
        #
        # Mapping:
        #   node_connections[i]     — L_i: list of node indices reachable from
        #                             node i along Voronoi boundary pixels.
        #                             This is the "correlated nodes table" used
        #                             directly by BPSO.decode_path (Eq. 6).
        #   distance_matrix[i][j]   — D(s_i, s_{j}): the number of boundary
        #                             pixels on the shortest path from node i
        #                             to node j, used in Eq. (7) to compute
        #                             the cost C(G_i).
        #   path_matrix[i][j]       — the pixel-level coordinates of the
        #                             boundary branch from node i to j; used
        #                             by the visualiser to draw the exact
        #                             Voronoi route (not in the paper, but
        #                             consistent with its description of
        #                             "non-smooth routes" between nodes).
        #   valid_pixels            — the set of cells the BFS may traverse:
        #                             boundary pixels ∪ node positions; this
        #                             confines path-finding to the Voronoi
        #                             network rather than open free space.
        #   BFS (deque)             — shortest-path search along boundary
        #                             pixels; BFS guarantees the minimum
        #                             pixel-count distance is found first,
        #                             which is the correct interpretation of
        #                             D(s_i, s_{i+1}) in Eq. (7).
        #   8-connected neighbours  — matches the paper's statement that "there
        #                             are 8 neighbouring grids in maximum for
        #                             each grid."
        # ---------------------------------------------------------------------
        """BFS from every node along boundary pixels to find reachable neighbours.

        Stores results in:
          self.node_connections  — adjacency lists  (correlated nodes table)
          self.distance_matrix   — pixel-path length between connected nodes
          self.path_matrix       — actual pixel coordinates of each path segment

        Performance fix: node membership is checked against a set (O(1)) rather
        than the original list (O(N) per check, called millions of times).
        """
        n_nodes = len(self.nodes)
        if n_nodes == 0:
            self.node_connections = {}
            return

        # Build a set of (row, col) for O(1) node lookup
        node_set: set = set(self.nodes)

        # Valid traversal pixels = boundaries ∪ nodes
        valid_pixels: set = set(self.voronoi_boundaries) | node_set

        # Initialise output structures
        self.node_connections = {i: [] for i in range(n_nodes)}
        self.distance_matrix = {i: {} for i in range(n_nodes)}
        self.path_matrix = {i: {} for i in range(n_nodes)}

        # Index: (row, col) --> node index
        node_index: Dict[Tuple[int, int], int] = {
            coord: idx for idx, coord in enumerate(self.nodes)
        }

        for i in range(n_nodes):
            start = self.nodes[i]
            queue: deque = deque([(start[0], start[1], [start])])
            visited: set = {start}

            while queue:
                r, c, current_path = queue.popleft()

                coord = (r, c)
                # -------------------------------------------------------------
                # [Paper §3.1.3 — correlated nodes table construction]
                # When BFS reaches a node different from the source, we have
                # found a direct Voronoi-boundary branch between two nodes.
                # Record: the connection in node_connections (L_i entry),
                #         the branch length in distance_matrix (D(i, j)),
                #         and the pixel route in path_matrix.
                # -------------------------------------------------------------
                # If we've reached a different node, record the connection
                if coord != start and coord in node_set:
                    j = node_index[coord]
                    if j not in self.node_connections[i]:
                        self.node_connections[i].append(j)
                        self.distance_matrix[i][j] = len(current_path) - 1
                        self.path_matrix[i][j] = current_path
                    # Don't traverse further through this node (BFS stops here)
                    continue

                # -------------------------------------------------------------
                # [Paper §3.1.2.2 — "8 neighbouring grids in maximum"]
                # Expand BFS to all 8-connected neighbours that lie on the
                # valid boundary/node pixel set and have not been visited yet.
                # -------------------------------------------------------------
                # 8-connected neighbours along valid pixels
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nb = (r + dr, c + dc)
                        if nb in valid_pixels and nb not in visited:
                            visited.add(nb)
                            queue.append((nb[0], nb[1], current_path + [nb]))

    def plot_obstacles(self):
        plt.figure(figsize=(6, 6))

        # Show obstacle map
        plt.imshow(self.obstacle_map, origin="lower")

        plt.colorbar(label="Obstacle ID")
        plt.title("Obstacle Map")
        plt.xlabel("X")
        plt.ylabel("Y")

        plt.savefig("debug_plots/avbn_plot_obstacles.png")
        plt.close()

    def plot_network(self) -> None:
        # Ensure network exists
        if not self.nodes:
            raise ValueError("Network not built. Call build_network() first.")

        plt.figure(figsize=(8, 8))

        # --- Background: obstacle map ---
        plt.imshow(self.obstacle_map, origin="lower", alpha=0.5)
        # plt.imshow(self.obstacle_map, origin="lower", cmap="gray", alpha=0.5)

        # --- Plot Voronoi boundaries ---
        if self.voronoi_boundaries:
            by = [p[0] for p in self.voronoi_boundaries]
            bx = [p[1] for p in self.voronoi_boundaries]
            plt.scatter(bx, by, s=1, color="blue", label="Boundaries")

        # --- Plot nodes ---
        node_x = [p[1] for p in self.nodes]
        node_y = [p[0] for p in self.nodes]
        plt.scatter(node_x, node_y, color="red", s=40, label="Nodes", zorder=3)

        # --- Plot connections (edges) ---
        for i in range(len(self.nodes)):
            for j in self.node_connections.get(i, []):
                if j in self.path_matrix.get(i, {}):
                    path = self.path_matrix[i][j]
                    px = [p[1] for p in path]
                    py = [p[0] for p in path]
                    plt.plot(px, py, color="green", linewidth=1)

        # --- Formatting ---
        plt.title("AVBN Network")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc="upper right")

        # --- Save figure ---
        os.makedirs("debug_plots", exist_ok=True)
        plt.savefig("debug_plots/avbn_plot_network.png", dpi=300, bbox_inches="tight")
        plt.close()

    def plot_obstacles_before_after(self, before: NDArray, after: NDArray) -> None:
        os.makedirs("debug_plots", exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # --- BEFORE ---
        axes[0].imshow(before, origin="lower")
        axes[0].set_title("Before Enlargement")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")

        # --- AFTER ---
        axes[1].imshow(after, origin="lower")
        axes[1].set_title("After Enlargement")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")

        plt.tight_layout()

        plt.savefig(
            "debug_plots/avbn_before_after_enlargement.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_voronoi_boundaries(self) -> None:
        boundaries = self.voronoi_boundaries.copy()

        plt.figure(figsize=(8, 8))

        # Background: Voronoi-labeled map
        plt.imshow(self.obstacle_map, origin="lower", cmap="tab20")

        # Overlay boundary points
        if boundaries:
            y = [p[0] for p in boundaries]
            x = [p[1] for p in boundaries]
            plt.scatter(x, y, s=1, color="red", label="Boundaries")

        plt.title("Voronoi Boundaries (Overlay)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()

        plt.savefig(
            "debug_plots/avbn_voronoi_boundaries_overlay.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_distance_matrix(self):
        n = len(self.nodes)
        mat = np.full((n, n), np.nan)

        for i in self.distance_matrix:
            for j in self.distance_matrix[i]:
                mat[i, j] = self.distance_matrix[i][j]

        plt.figure(figsize=(6, 6))
        plt.imshow(mat)
        plt.colorbar(label="Distance")
        plt.title("Distance Matrix")
        plt.xlabel("Node")
        plt.ylabel("Node")
        plt.savefig("debug_plots/avbn_distance_matrix.png")
        plt.close()

    def plot_correlated_nodes_table(self) -> None:

        if not self.node_connections:
            raise ValueError("Node connections not built. Call build_network() first.")

        data = []
        for node, neighbors in self.node_connections.items():
            data.append(
                {
                    "Node": node,
                    "L_j (Connected Nodes)": str(neighbors),
                    "l_j (Count)": len(neighbors),
                }
            )

        df = pd.DataFrame(data)

        # Plot as a table
        fig, ax = plt.subplots(figsize=(8, len(df) * 0.5 + 2))
        ax.axis("off")

        table = ax.table(
            cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)

        plt.title("Correlated Nodes Table (L_j)")
        plt.savefig("debug_plots/avbn_correlated_nodes_table.png", bbox_inches="tight")
        plt.close()
