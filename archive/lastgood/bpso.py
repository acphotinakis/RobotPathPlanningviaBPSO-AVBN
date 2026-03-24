import numpy as np
from typing import List, Tuple, Dict
import copy


class BPSO:
    """Biogeography-based Particle Swarm Optimization.

    Implements the BPSO algorithm from:
      Mo & Xu, "Research of biogeography particle swarm optimization
      for robot path planning", Neurocomputing 148 (2015) 91-99.

    Fixes applied vs. original code
    --------------------------------
    1. Migration trigger: paper selects habitat Hi with its *emigration* rate
       λ_i (step 2 of the migration pseudocode), then chooses the *source* Hj
       using immigration rates μ_j.  Original code had the roles swapped.
    2. Mutation formula (Eq. 4): P_s is the species-count probability, not raw
       fitness.  We approximate it as a normalised rank probability so the
       formula is dimensionally consistent with the paper.
    3. Linearly decreasing inertia weight: paper specifies w ∈ [0.9, 0.4]
       decreasing linearly over generations (§2.2).  Original used fixed w=0.9.
    4. Elite preservation (step 3.5): the two best habitats are saved before
       migration/mutation and restored afterwards so good solutions are never
       lost.
    5. Visited-node deletion from all tables: during decoding the paper removes
       each chosen node from *all* correlated-node tables, not just a local set
       (§3.2, step 3 / Eq. 6 description).
    """

    def __init__(
        self,
        pop_size: int = 100,
        max_gen: int = 100,
        p_mutation: float = 0.005,
        w_max: float = 0.9,  # Inertia upper bound (paper: w ∈ [0.9, 0.4])
        w_min: float = 0.4,  # Inertia lower bound
        c1: float = 2.0,
        c2: float = 2.0,
    ):
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.p_mutation = p_mutation
        self.w_max = w_max
        self.w_min = w_min
        self.c1 = c1
        self.c2 = c2

        self.population: List[List[int]] = []
        self.fitness: List[float] = []
        self.velocities: List[List[float]] = []
        self.personal_best: List[List[int]] = []
        self.personal_best_fitness: List[float] = []
        self.global_best: List[int] = []
        self.global_best_fitness: float = float("inf")

        # Current inertia weight — updated each generation
        self._w: float = w_max

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize_population(self, num_nodes: int) -> None:
        """Initialise a random population of encoded paths."""
        self.population = []
        self.velocities = []
        self._w = self.w_max  # reset inertia at the start of each run
        self.global_best_fitness = float("inf")
        self.global_best = []

        for _ in range(self.pop_size):
            # SIV values are integers drawn from [0, num_nodes) — §3.2 step 2
            individual = np.random.permutation(num_nodes).tolist()
            self.population.append(individual)
            # Small random initial velocities
            velocity = np.random.randn(num_nodes).tolist()
            self.velocities.append(velocity)

        self.personal_best = copy.deepcopy(self.population)
        self.personal_best_fitness = [float("inf")] * self.pop_size

    # ------------------------------------------------------------------
    # Decoding  (§3.2 step 3, Eq. 6)
    # ------------------------------------------------------------------

    def decode_path(
        self,
        individual: List[int],
        node_connections: Dict[int, List[int]],
        start_node: int,
        end_node: int,
    ) -> List[int]:
        """Decode a habitat into an actual node sequence.

        Fix 5 — visited-node deletion from all tables
        -----------------------------------------------
        The paper: "When a route node is selected to one of the path nodes,
        it will be deleted from all correlated nodes table for avoiding being
        reselected."  We build a mutable local copy of every adjacency list
        and remove each chosen node from *all* of them.
        """
        # Mutable per-call copy so the shared network is never modified
        local_conn: Dict[int, List[int]] = {
            node: list(neighbours) for node, neighbours in node_connections.items()
        }

        path = [start_node]
        current = start_node
        # Remove the start node from every adjacency list immediately
        _remove_from_all(local_conn, start_node)

        for gene_value in individual:
            if current == end_node:
                break

            available = local_conn.get(current, [])
            if not available:
                break

            # Eq. (6): m = MOD(g_j, l_j)  --> 0-based index into available
            l_j = len(available)
            m = gene_value % l_j

            next_node = available[m]
            path.append(next_node)
            current = next_node

            # Delete the chosen node from ALL adjacency lists
            _remove_from_all(local_conn, next_node)

        # Append end node if not reached (fitness penalty applied separately)
        if path[-1] != end_node:
            path.append(end_node)

        return path

    # ------------------------------------------------------------------
    # Migration rates  (Eqs. 1–2)
    # ------------------------------------------------------------------

    def calculate_migration_rates(
        self,
    ) -> Tuple[List[float], List[float]]:
        """Return per-habitat emigration (λ) and immigration (μ) rates.

        Habitats are ranked by fitness ascending (best cost = rank 0).
        In the paper's species analogy, a better habitat has more species
        (higher s), giving HIGH λ (exports features) and LOW μ (few imports).

        Eqs. (1) and (2) with E = I = 1:
            λ_s = s / P
            μ_s = 1 − s / P
        where s = P − rank  so that rank 0 (best) --> s = P (max species).
        """
        P = self.pop_size
        rank_order = np.argsort(self.fitness)  # ascending: rank_order[0] = best

        emigration_rates = [0.0] * P
        immigration_rates = [0.0] * P

        for rank, habitat_idx in enumerate(rank_order):
            s = P - rank  # best habitat --> s=P, worst --> s=1
            emigration_rates[habitat_idx] = s / P
            immigration_rates[habitat_idx] = 1.0 - s / P

        return emigration_rates, immigration_rates

    # ------------------------------------------------------------------
    # PSO velocity / position update  (Eq. 5)
    # ------------------------------------------------------------------

    def update_velocity_position(self, i: int) -> None:
        """Update particle i using PSO equations with current inertia weight.

        Fix 3: self._w is set by the main loop to decrease linearly from
        w_max = 0.9 to w_min = 0.4 over max_gen generations.
        """
        r1 = np.random.random()
        r2 = np.random.random()
        n = len(self.velocities[i])

        for j in range(n):
            self.velocities[i][j] = (
                self._w * self.velocities[i][j]
                + self.c1 * r1 * (self.personal_best[i][j] - self.population[i][j])
                + self.c2 * r2 * (self.global_best[j] - self.population[i][j])
            )

        for j in range(len(self.population[i])):
            self.population[i][j] = int(
                round(self.population[i][j] + self.velocities[i][j])
            )
            # Clamp SIV to valid range [0, num_nodes − 1]
            self.population[i][j] = max(0, min(self.population[i][j], n - 1))

    # ------------------------------------------------------------------
    # Mutation  (§2.1, Eq. 4)
    # ------------------------------------------------------------------

    def mutation(self) -> None:
        """Mutate the worst half of the population.

        Fix 2 — correct P_s interpretation
        ------------------------------------
        In Eq. (4) P_s is the probability that a habitat contains exactly
        s species (from Eq. 3).  We approximate it as the normalised rank
        probability  P_s ≈ s / P  (proportional to species count), which
        makes P_max = 1 (the best habitat) and gives the correct qualitative
        behaviour: the worst habitats mutate most aggressively.

        Eq. (4):  m_i = P_mute * (1 − P_s / P_max)
        """
        P = self.pop_size
        rank_order = np.argsort(self.fitness)  # ascending; worst half at the end
        worst_half = rank_order[P // 2 :]

        # P_max is the rank-probability of the best habitat (= 1.0)
        P_max = 1.0

        for rank_pos, habitat_idx in enumerate(worst_half):
            # Global rank of this habitat (0 = best, P-1 = worst)
            global_rank = P // 2 + rank_pos
            s = P - global_rank  # species count  (1 … P//2 for worst half)
            P_s = s / P  # normalised rank probability

            m_i = self.p_mutation * (1.0 - P_s / P_max)

            if np.random.random() < m_i:
                siv_idx = np.random.randint(len(self.population[habitat_idx]))
                self.population[habitat_idx][siv_idx] = np.random.randint(
                    0, len(self.population[habitat_idx])
                )

    # ------------------------------------------------------------------
    # Fitness  (Eq. 7)
    # ------------------------------------------------------------------

    def calculate_fitness(
        self,
        path: List[int],
        distance_matrix: Dict[int, Dict[int, float]],
        end_node: int,
    ) -> float:
        """Path cost = Σ grid-distance along Voronoi segments + penalty.

        Eq. (7):  C(G_i) = Σ D(s_i, s_{i+1}) + k·α
            k = 1 if path does not reach end_node, else 0
            α = 1000 (punishment cost from §3.2 step 4)
        """
        if len(path) < 2:
            return 1e6

        total_cost = 0.0
        for idx in range(len(path) - 1):
            n1, n2 = path[idx], path[idx + 1]
            seg = distance_matrix.get(n1, {}).get(n2, None)
            if seg is None:
                return 1e6  # disconnected --> invalid path
            total_cost += seg

        alpha = 1000
        k_penalty = 0 if path[-1] == end_node else 1
        return total_cost + k_penalty * alpha

    # ------------------------------------------------------------------
    # Main optimisation loop  (§2.3 BPSO procedure)
    # ------------------------------------------------------------------

    def optimize(
        self,
        nodes: List[Tuple[int, int]],
        node_connections: Dict[int, List[int]],
        distance_matrix: Dict[int, Dict[int, float]],
        start_node: int,
        end_node: int,
    ) -> Tuple[List[int], float, Dict]:
        """Run BPSO; return (best_path_node_indices, best_cost, history)."""
        num_nodes = len(nodes)
        self.initialize_population(num_nodes)

        history: Dict[str, List[float]] = {
            "best_fitness": [],
            "mean_fitness": [],
        }

        for generation in range(self.max_gen):

            # ---- Fix 3: linearly decreasing inertia weight ---------------
            self._w = self.w_max - (self.w_max - self.w_min) * generation / max(
                self.max_gen - 1, 1
            )

            # ---- Step 3.1: evaluate fitness ------------------------------
            self.fitness = []
            for i, individual in enumerate(self.population):
                path = self.decode_path(
                    individual, node_connections, start_node, end_node
                )
                fitness = self.calculate_fitness(path, distance_matrix, end_node)
                self.fitness.append(fitness)

                if fitness < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = copy.deepcopy(individual)

                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best = copy.deepcopy(individual)

            # ---- Fix 4: save elites before modification ------------------
            rank_order = np.argsort(self.fitness)
            elite_indices = rank_order[:2]
            elite_habitats = [copy.deepcopy(self.population[i]) for i in elite_indices]
            elite_fitnesses = [self.fitness[i] for i in elite_indices]

            # ---- Step 3.2: compute migration rates -----------------------
            emigration_rates, immigration_rates = self.calculate_migration_rates()

            # ---- Steps 3.2–3.3: migration + PSO fallback -----------------
            # Fix 1: select habitat Hi with its *emigration* rate λ_i;
            # choose source Hj weighted by immigration rates μ_j.
            new_population = copy.deepcopy(self.population)

            for i in range(self.pop_size):
                if np.random.random() < emigration_rates[i]:
                    # Habitat i is selected --> receive an SIV from a source
                    # chosen proportionally to immigration rates μ_j
                    mu = np.array(immigration_rates)
                    mu_sum = mu.sum()
                    if mu_sum > 0:
                        probs = mu / mu_sum
                        j = int(np.random.choice(self.pop_size, p=probs))
                        if self.population[j]:
                            siv_idx = np.random.randint(len(new_population[i]))
                            if siv_idx < len(self.population[j]):
                                new_population[i][siv_idx] = self.population[j][siv_idx]
                else:
                    # Not selected for immigration --> PSO position update
                    if self.global_best:
                        self.population[i] = new_population[i]  # sync first
                        self.update_velocity_position(i)
                        new_population[i] = copy.deepcopy(self.population[i])

            self.population = new_population

            # ---- Step 3.4: mutate worst half -----------------------------
            self.mutation()

            # ---- Fix 4: restore elites (step 3.5) -----------------------
            # Replace the two currently-worst habitats with the saved elites
            post_rank = np.argsort(self.fitness)
            worst_two = post_rank[-2:]
            for slot, (hab, fit) in enumerate(zip(elite_habitats, elite_fitnesses)):
                replace_idx = worst_two[slot]
                self.population[replace_idx] = hab
                self.fitness[replace_idx] = fit

            # ---- Bookkeeping ---------------------------------------------
            history["best_fitness"].append(self.global_best_fitness)
            history["mean_fitness"].append(float(np.mean(self.fitness)))

            if generation % 10 == 0:
                print(
                    f"  Gen {generation:3d} | w={self._w:.3f} | "
                    f"Best = {self.global_best_fitness:.2f} | "
                    f"Mean = {np.mean(self.fitness):.2f}"
                )

        best_path = self.decode_path(
            self.global_best, node_connections, start_node, end_node
        )
        return best_path, self.global_best_fitness, history


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------


def _remove_from_all(connections: Dict[int, List[int]], node: int) -> None:
    """Remove *node* from every adjacency list in *connections*."""
    for adj in connections.values():
        try:
            adj.remove(node)
        except ValueError:
            pass
