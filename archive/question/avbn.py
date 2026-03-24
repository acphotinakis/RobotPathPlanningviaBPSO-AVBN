import numpy as np
from typing import List, Tuple, Dict


class AVBN:
    """Approximate Voronoi Boundary Network for environment modeling"""

    def __init__(self, grid_size: Tuple[int, int] = (400, 400)):
        self.grid_size = grid_size
        self.obstacle_map = np.zeros(grid_size, dtype=int)
        self.voronoi_boundaries = []
        self.nodes = []
        self.node_connections = {}
        self.distance_matrix = {}
        self.path_matrix = {}

    def add_obstacles(self, obstacles: List[Tuple[int, int, int, int]]):
        """Add rectangular obstacles to the map
        Args:
            obstacles: List of (x, y, width, height) tuples
        """
        for idx, (x, y, w, h) in enumerate(obstacles, start=1):
            self.obstacle_map[y : y + h, x : x + w] = idx

    def enlarge_obstacles(self):
        """Enlarge obstacles until all free space is filled (Voronoi partitioning)"""
        from scipy.ndimage import binary_dilation

        # The enlargement should not stop until all the free grids are fused into eroded area
        while (self.obstacle_map == 0).any():
            new_map = self.obstacle_map.copy()
            for i in range(1, self.obstacle_map.max() + 1):
                mask = self.obstacle_map == i
                dilated = binary_dilation(mask)
                # Grow into free space without overwriting other growing obstacles
                new_cells = dilated & (self.obstacle_map == 0) & (new_map == 0)
                new_map[new_cells] = i
            self.obstacle_map = new_map

    def compute_distance_field(self):
        """Compute distance from each free cell to nearest obstacle"""
        from scipy.ndimage import distance_transform_edt

        free_space = self.obstacle_map == 0
        distance_field = distance_transform_edt(free_space)
        return distance_field

    def find_voronoi_boundaries(self):
        """Find Voronoi boundaries where different areas meet"""
        boundaries = []
        h, w = self.grid_size

        for i in range(h - 1):
            for j in range(w - 1):
                val = self.obstacle_map[i, j]
                # A grid will be labeled as a boundary grid if the upside grid or right grid is not at the same value as itself
                if (
                    val != self.obstacle_map[i + 1, j]
                    or val != self.obstacle_map[i, j + 1]
                ):
                    boundaries.append((i, j))

        return boundaries

    def find_nodes(self, boundaries: List[Tuple[int, int]]):
        """Find intersection nodes in the boundary network"""
        nodes = []

        for i, j in boundaries:
            obstacle_ids = set()
            # Check 8-neighborhood
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
                        obstacle_ids.add(self.obstacle_map[ni, nj])

            # A boundary grid and its neighboring grids are integrated as one Voronoi node if the neighboring grids belong to three or more than three different areas
            if len(obstacle_ids) >= 3:
                nodes.append((i, j))

        # Filter: Cluster nodes that are too close to each other to form single intersection points
        if not nodes:
            return []

        filtered_nodes = []
        for node in nodes:
            if not filtered_nodes:
                filtered_nodes.append(node)
            else:
                dist = np.min(
                    np.sqrt(
                        np.sum((np.array(filtered_nodes) - np.array(node)) ** 2, axis=1)
                    )
                )
                if (
                    dist > 15
                ):  # Distance threshold to merge nodes at the same intersection
                    filtered_nodes.append(node)

        return filtered_nodes

    def build_network(self):
        """Build the complete AVBN network"""
        # Find boundaries and nodes
        boundaries = self.find_voronoi_boundaries()
        self.voronoi_boundaries = boundaries
        self.nodes = self.find_nodes(boundaries)

        # Build node connections using simple connectivity
        self.node_connections = self._build_connections()

        return self.nodes, self.distance_matrix

    def _build_connections(self):
        """Build connections and compute grid-based distances between nodes along Voronoi boundaries"""
        from collections import deque

        connections = {i: [] for i in range(len(self.nodes))}
        self.distance_matrix = {}  # Store exact grid distances for fitness calculation

        # Create a fast lookup set for boundary pixels and nodes
        valid_path_pixels = set(self.voronoi_boundaries)
        for node in self.nodes:
            valid_path_pixels.add(node)

        for i in range(len(self.nodes)):
            self.distance_matrix[i] = {}
            start_r, start_c = self.nodes[i]

            # BFS queue: (row, col, path_so_far)
            queue = deque([(start_r, start_c, [(start_r, start_c)])])
            visited = {(start_r, start_c)}

            self.path_matrix[i] = {}  # Initialize path storage for node i

            while queue:
                r, c, current_path = queue.popleft()

                # Check if we reached another node
                if (r, c) != (start_r, start_c) and (r, c) in self.nodes:
                    j = self.nodes.index((r, c))
                    if j not in connections[i]:
                        connections[i].append(j)
                        self.distance_matrix[i][j] = len(current_path) - 1
                        self.path_matrix[i][
                            j
                        ] = current_path  # Store the pixel coordinates
                    continue

                # 8-way connectivity
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc

                        if (nr, nc) in valid_path_pixels and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            # Pass the updated path history forward
                            queue.append((nr, nc, current_path + [(nr, nc)]))

        return connections
