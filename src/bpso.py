import numpy as np
from typing import List, Tuple, Dict
import copy


class BPSO:
    """Biogeography-based Particle Swarm Optimization"""

    def __init__(
        self,
        pop_size: int = 100,
        max_gen: int = 100,
        p_mutation: float = 0.005,
        w: float = 0.9,
        c1: float = 2.0,
        c2: float = 2.0,
    ):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.p_mutation = p_mutation
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.population = []
        self.fitness = []
        self.velocities = []
        self.personal_best = []
        self.personal_best_fitness = []
        self.global_best = None
        self.global_best_fitness = float("inf")

    def initialize_population(self, num_nodes: int):
        """Initialize random population of paths"""
        self.population = []
        self.velocities = []

        for _ in range(self.pop_size):
            # Random permutation of node indices
            individual = np.random.permutation(num_nodes).tolist()
            self.population.append(individual)

            # Initialize velocity
            velocity = np.random.randn(num_nodes).tolist()
            self.velocities.append(velocity)

        self.personal_best = copy.deepcopy(self.population)
        self.personal_best_fitness = [float("inf")] * self.pop_size

    def decode_path(
        self,
        individual: List[int],
        node_connections: Dict,
        start_node: int,
        end_node: int,
    ) -> List[int]:
        """Decode individual to actual path using Eq. (6) from paper

        Args:
            individual: Encoded solution (habitat)
            node_connections: Dictionary of node connections
            start_node: Starting node index
            end_node: Ending node index
        """
        path = [start_node]
        current = start_node
        visited = {start_node}

        for gene_value in individual:
            if current == end_node:
                break

            # Get available connections
            available = [
                n for n in node_connections.get(current, []) if n not in visited
            ]

            if not available:
                break

            # Apply decoding rule: m = MOD(g_j, l_j) + 1
            l_j = len(available)
            m = gene_value % l_j

            next_node = available[m]
            path.append(next_node)
            visited.add(next_node)
            current = next_node

        # Add end node if not reached
        if path[-1] != end_node and end_node not in visited:
            path.append(end_node)

        return path

    def calculate_fitness(
        self, path: List[int], nodes: List[Tuple[int, int]], obstacle_map: np.ndarray
    ) -> float:
        """Calculate path cost (Eq. 7 from paper)"""
        if len(path) < 2:
            return 1e6  # Invalid path

        total_cost = 0

        for i in range(len(path) - 1):
            if path[i] >= len(nodes) or path[i + 1] >= len(nodes):
                return 1e6

            p1 = nodes[path[i]]
            p2 = nodes[path[i + 1]]

            # Euclidean distance
            dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            total_cost += dist

        # Add penalty if path is invalid
        if path[-1] != len(nodes) - 1:
            total_cost += 1000  # Penalty parameter α

        return total_cost

    def calculate_migration_rates(self):
        """Calculate emigration and immigration rates (Eqs. 1-2)"""
        # Sort by fitness
        sorted_indices = np.argsort(self.fitness)

        E = 1.0  # Max emigration rate
        I = 1.0  # Max immigration rate

        emigration_rates = []
        immigration_rates = []

        for idx in sorted_indices:
            s = list(sorted_indices).index(idx) + 1  # Species count (rank)

            # Eq. (1): λ_s = E * s / P
            lambda_s = E * s / self.pop_size

            # Eq. (2): μ_s = I * (1 - s/P)
            mu_s = I * (1 - s / self.pop_size)

            emigration_rates.append(lambda_s)
            immigration_rates.append(mu_s)

        return emigration_rates, immigration_rates

    def migration(self, emigration_rates: List[float], immigration_rates: List[float]):
        """Perform BBO migration operation"""
        new_population = copy.deepcopy(self.population)

        for i in range(self.pop_size):
            # Select habitat for immigration with probability λ_i
            if np.random.random() < emigration_rates[i]:
                # Select source habitat based on immigration rates
                probabilities = np.array(immigration_rates)
                probabilities = probabilities / probabilities.sum()
                j = np.random.choice(self.pop_size, p=probabilities)

                # Randomly select SIV to migrate
                if len(self.population[j]) > 0:
                    siv_idx = np.random.randint(len(self.population[j]))
                    new_population[i][siv_idx] = self.population[j][siv_idx]

        self.population = new_population

    def update_velocity_position(self, i: int):
        """Update particle using PSO equations (Eq. 5)"""
        # Update velocity
        r1 = np.random.random()
        r2 = np.random.random()

        for j in range(len(self.velocities[i])):
            self.velocities[i][j] = (
                self.w * self.velocities[i][j]
                + self.c1 * r1 * (self.personal_best[i][j] - self.population[i][j])
                + self.c2 * r2 * (self.global_best[j] - self.population[i][j])
            )

        # Update position
        for j in range(len(self.population[i])):
            self.population[i][j] = int(self.population[i][j] + self.velocities[i][j])
            # Keep within bounds
            self.population[i][j] = max(
                0, min(self.population[i][j], len(self.population[i]) - 1)
            )

    def mutation(self):
        """Perform mutation on worst half of population (Eq. 4)"""
        sorted_indices = np.argsort(self.fitness)
        worst_half = sorted_indices[self.pop_size // 2 :]

        # Calculate P_max
        fitness_array = np.array(self.fitness)
        P_max = np.max(fitness_array)

        for idx in worst_half:
            # Eq. (4): m_i = P_mute * (1 - P_s / P_max)
            P_s = fitness_array[idx]
            m_i = self.p_mutation * (1 - P_s / P_max) if P_max > 0 else self.p_mutation

            if np.random.random() < m_i:
                # Mutate random SIV
                siv_idx = np.random.randint(len(self.population[idx]))
                self.population[idx][siv_idx] = np.random.randint(
                    0, len(self.population[idx])
                )

    def optimize(
        self,
        nodes: List[Tuple[int, int]],
        node_connections: Dict,
        obstacle_map: np.ndarray,
        start_node: int,
        end_node: int,
    ):
        """Main BPSO optimization loop"""
        num_nodes = len(nodes)
        self.initialize_population(num_nodes)

        history = {"best_fitness": [], "mean_fitness": []}

        for generation in range(self.max_gen):
            # Evaluate fitness
            self.fitness = []
            for i, individual in enumerate(self.population):
                path = self.decode_path(
                    individual, node_connections, start_node, end_node
                )
                fitness = self.calculate_fitness(path, nodes, obstacle_map)
                self.fitness.append(fitness)

                # Update personal best
                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = copy.deepcopy(individual)

                # Update global best
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = copy.deepcopy(individual)

            # Calculate migration rates
            emigration_rates, immigration_rates = self.calculate_migration_rates()

            # Migration and PSO update
            for i in range(self.pop_size):
                if np.random.random() < immigration_rates[i]:
                    # Perform migration
                    self.migration(emigration_rates, immigration_rates)
                else:
                    # Update using PSO
                    if self.global_best is not None:
                        self.update_velocity_position(i)

            # Mutation
            self.mutation()

            # Track progress
            history["best_fitness"].append(self.global_best_fitness)
            history["mean_fitness"].append(np.mean(self.fitness))

            if generation % 10 == 0:
                print(
                    f"Generation {generation}: Best fitness = {self.global_best_fitness:.2f}"
                )

        # Decode best path
        best_path = self.decode_path(
            self.global_best, node_connections, start_node, end_node
        )

        return best_path, self.global_best_fitness, history
