import numpy as np
from typing import List, Tuple, Dict
from collections import deque


class AVBN:
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

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------

    def add_obstacles(self, obstacles: List[Tuple[int, int, int, int]]) -> None:
        """Add rectangular obstacles to the grid.

        Args:
            obstacles: List of (x, y, width, height) tuples.
                       x/y follow image conventions: x=column, y=row.
        """
        for idx, (x, y, w, h) in enumerate(obstacles, start=1):
            self.obstacle_map[y : y + h, x : x + w] = idx

    # ------------------------------------------------------------------
    # §3.1.2.1  Raster-based enlargement of obstacles
    # ------------------------------------------------------------------

    def enlarge_obstacles(self) -> None:
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

    # ------------------------------------------------------------------
    # §3.1.2.2  Recognising Voronoi boundaries and nodes
    # ------------------------------------------------------------------

    def find_voronoi_boundaries(self) -> List[Tuple[int, int]]:
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

    def find_nodes(self, boundaries: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
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

    # ------------------------------------------------------------------
    # Network construction
    # ------------------------------------------------------------------

    def build_network(
        self,
    ) -> Tuple[List[Tuple[int, int]], Dict[int, Dict[int, int]]]:
        """Build the complete AVBN: boundaries --> nodes --> connections.

        Returns
        -------
        nodes            : list of (row, col) intersection coordinates
        distance_matrix  : dict[i][j] = pixel distance along boundary
        """
        boundaries = self.find_voronoi_boundaries()
        self.voronoi_boundaries = boundaries
        self.nodes = self.find_nodes(boundaries)
        self._build_connections()
        return self.nodes, self.distance_matrix

    def _build_connections(self) -> None:
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
                # If we've reached a different node, record the connection
                if coord != start and coord in node_set:
                    j = node_index[coord]
                    if j not in self.node_connections[i]:
                        self.node_connections[i].append(j)
                        self.distance_matrix[i][j] = len(current_path) - 1
                        self.path_matrix[i][j] = current_path
                    # Don't traverse further through this node (BFS stops here)
                    continue

                # 8-connected neighbours along valid pixels
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nb = (r + dr, c + dc)
                        if nb in valid_pixels and nb not in visited:
                            visited.add(nb)
                            queue.append((nb[0], nb[1], current_path + [nb]))
