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

    def add_obstacles(self, obstacles: List[Tuple[int, int, int, int]]):
        """Add rectangular obstacles to the map
        Args:
            obstacles: List of (x, y, width, height) tuples
        """
        for idx, (x, y, w, h) in enumerate(obstacles, start=1):
            self.obstacle_map[y : y + h, x : x + w] = idx

    # def enlarge_obstacles(self, iterations: int = 5):
    #     """Enlarge obstacles to account for robot size and safety margin"""
    #     from scipy.ndimage import binary_dilation

    #     for i in range(1, self.obstacle_map.max() + 1):
    #         mask = self.obstacle_map == i
    #         for _ in range(iterations):
    #             dilated = binary_dilation(mask)
    #             new_cells = dilated & (self.obstacle_map == 0)
    #             self.obstacle_map[new_cells] = i
    #             mask = dilated

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

    # def find_voronoi_boundaries(self):
    #     """Find Voronoi boundaries between obstacles"""
    #     boundaries = []
    #     h, w = self.grid_size

    #     for i in range(h):
    #         for j in range(w):
    #             if self.obstacle_map[i, j] == 0:
    #                 # Check 8-connected neighbors
    #                 neighbors = []
    #                 for di in [-1, 0, 1]:
    #                     for dj in [-1, 0, 1]:
    #                         if di == 0 and dj == 0:
    #                             continue
    #                         ni, nj = i + di, j + dj
    #                         if 0 <= ni < h and 0 <= nj < w:
    #                             if self.obstacle_map[ni, nj] > 0:
    #                                 neighbors.append(self.obstacle_map[ni, nj])

    #                 # If adjacent to 2+ different obstacles, it's a boundary
    #                 if len(set(neighbors)) >= 2:
    #                     boundaries.append((i, j))

    #     return boundaries

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

    # def find_nodes(self, boundaries: List[Tuple[int, int]]):
    #     """Find intersection nodes in the boundary network"""
    #     nodes = []
    #     boundary_set = set(boundaries)

    #     for i, j in boundaries:
    #         # Count different obstacle regions in 8-neighborhood
    #         obstacle_ids = set()
    #         for di in [-1, 0, 1]:
    #             for dj in [-1, 0, 1]:
    #                 ni, nj = i + di, j + dj
    #                 if 0 <= ni < self.grid_size[0] and 0 <= nj < self.grid_size[1]:
    #                     if self.obstacle_map[ni, nj] > 0:
    #                         obstacle_ids.add(self.obstacle_map[ni, nj])

    #         # Node if adjacent to 3+ obstacles
    #         if len(obstacle_ids) >= 3:
    #             nodes.append((i, j))

    #     return nodes

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

        return self.nodes, self.node_connections

    def _build_connections(self):
        """Build connections between nodes"""
        connections = {i: [] for i in range(len(self.nodes))}

        # Simple distance-based connections
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                dist = np.sqrt(
                    (self.nodes[i][0] - self.nodes[j][0]) ** 2
                    + (self.nodes[i][1] - self.nodes[j][1]) ** 2
                )

                # Connect if reasonably close (heuristic)
                if dist < 100:  # Adjust threshold as needed
                    connections[i].append(j)
                    connections[j].append(i)

        return connections
