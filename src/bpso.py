# =============================================================================
# bpso.py — Biogeography-based Particle Swarm Optimization
# =============================================================================
#
# PAPER REFERENCE
# ---------------
# Mo, H. & Xu, L. (2015). "Research of biogeography particle swarm
# optimization for robot path planning." Neurocomputing, 148, 91-99.
# https://doi.org/10.1016/j.neucom.2012.07.060
#
# This file implements the BPSO algorithm described in Section 2 of the paper,
# and the AVBN-based path-planning procedure described in Section 3.2.
#
# HOW TO READ THE ANNOTATIONS
# ----------------------------
# Each annotation block immediately precedes the code it describes and
# follows a three-part format:
#
#   [Paper §X.Y / Eq. N]  — location in the paper
#   Quote: "..."          — verbatim sentence(s) from the paper
#   Mapping:              — how paper symbols map to Python variables
#
# =============================================================================


import numpy as np
from typing import List, Tuple, Dict
import copy


class BPSO:
    # -------------------------------------------------------------------------
    # [Paper §1 Introduction / §2 Overview]
    # Quote: "This paper presents a new method of global path planning by
    #         combining BBO, PSO and approximate voronoi boundary network
    #         (AVBN) in a static environment."
    # Mapping: This class is the direct implementation of the hybrid BPSO
    #          algorithm. The BBO migration operator (Section 2.1) and the PSO
    #          position-update rule (Section 2.2) are both encapsulated here,
    #          with the AVBN network (avbn.py) supplying the graph that BPSO
    #          searches over.
    # -------------------------------------------------------------------------

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
        # ---------------------------------------------------------------------
        # [Paper §2.2 PSO / §4.2 Parameters setting]
        # Quote: "PSO: inertia w=0.9, c1 and c2=2."
        # Quote (§2.2): "w is the inertia in range [0.9, 0.4] that increases
        #                linearly."  (Note: the paper says 'increases' but the
        #                schedule runs from 0.9 down to 0.4 over generations,
        #                consistent with standard linearly-decreasing inertia.)
        # Mapping:
        #   w_max  = upper bound of inertia weight w   → paper value 0.9
        #   w_min  = lower bound of inertia weight w   → paper value 0.4
        #   c1, c2 = cognitive and social acceleration constants → paper value 2
        #   p_mutation = P_mute, the base mutation-rate parameter → paper 0.005
        # ---------------------------------------------------------------------
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

        # ---------------------------------------------------------------------
        # [Paper §2.1 BBO / §2.3 BPSO procedure — Step 3.1]
        # Quote: "In BBO, a population of candidate solutions is represented as
        #         vectors of integers. Each integer in the solution vector is
        #         considered to be an SIV."
        # Mapping:
        #   self.population          — the population P; each element is a
        #                              habitat G_i (a vector of SIV integers)
        #   self.fitness             — HSI (Habitat Suitability Index) per
        #                              habitat; lower path cost = higher HSI
        #   self.velocities          — V_i from PSO Eq. (5); one velocity
        #                              vector per particle/habitat
        #   self.personal_best       — P_i in Eq. (5): each particle's
        #                              historically best position
        #   self.personal_best_fitness — the HSI value at personal_best[i]
        #   self.global_best         — P_g in Eq. (5): best position found
        #                              across the entire swarm
        #   self.global_best_fitness — HSI at global_best
        # ---------------------------------------------------------------------

        self.population: List[List[int]] = []
        self.fitness: List[float] = []
        self.velocities: List[List[float]] = []
        self.personal_best: List[List[int]] = []
        self.personal_best_fitness: List[float] = []
        self.global_best: List[int] = []
        self.global_best_fitness: float = float("inf")

        # Current inertia weight — updated each generation
        self._w: float = w_max

    # =========================================================================
    # SECTION 2.3 — BPSO Step 1: Initialisation
    # =========================================================================

    def initialize_population(self, num_nodes: int) -> None:
        # ---------------------------------------------------------------------
        # [Paper §3.2 — Step 2: Encoding / §2.3 BPSO — Step 1]
        # Quote: "SIV value of a habitat G_i is any integral number of Q,
        #         where G_j = {g1, g2, ...}, g_j ∈ Q, j ∈ [1, 15]."
        # Quote: "Initialize parameters: P, G, P_mute, w, c1, c2."
        # Mapping:
        #   num_nodes   — |Q|, the total number of route nodes in the AVBN
        #                 network. In the paper's Fig. 2 example, |Q| = 15.
        #   individual  — one habitat G_i; each entry g_j is an integer drawn
        #                 uniformly from [0, num_nodes), satisfying g_j ∈ Q.
        #   velocity    — the initial velocity vector V_i^0 for particle i,
        #                 drawn from a standard normal distribution (small
        #                 random values ensure no single direction dominates
        #                 the first PSO update).
        # ---------------------------------------------------------------------

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

    # =========================================================================
    # SECTION 3.2 — Step 3: Decoding (Equation 6)
    # =========================================================================

    def decode_path(
        self,
        individual: List[int],
        node_connections: Dict[int, List[int]],
        start_node: int,
        end_node: int,
    ) -> List[int]:
        # ---------------------------------------------------------------------
        # [Paper §3.2 — Step 3 / Eq. (6)]
        # Quote: "In this step, the value g_j of SIV is mapped to a route
        #         node. Here, L_j[m] is the m-th node of L_j in correlated
        #         nodes table. Assuming s_j is a node selected to be one of
        #         the route nodes, then the next node is: s_{j+1} = L_j[m],
        #         where m is the sequence number in L_j; and the decoding
        #         rule is  m = MOD(g_j, l_j) + 1."
        # Quote: "When a route node is selected to one of the path nodes, it
        #         will be deleted from all correlated nodes table for avoiding
        #         being reselected."
        # Mapping:
        #   individual          — habitat G_i; a sequence of SIV integers g_j
        #   gene_value          — g_j: the j-th SIV in the habitat
        #   local_conn          — the mutable copy of L_j (correlated nodes
        #                         table); each entry local_conn[n] is the list
        #                         of nodes reachable from node n (= L_j in
        #                         the paper)
        #   available           — L_j, the candidate list for the current node
        #   l_j                 — l_j = |L_j|: length of the candidate list
        #   m                   — m = MOD(g_j, l_j): 0-based index into L_j
        #                         (paper uses 1-based; we subtract 1 implicitly
        #                         by using Python's 0-based list indexing)
        #   next_node           — s_{j+1} = L_j[m]: the selected next node
        #   _remove_from_all()  — implements "deleted from all correlated
        #                         nodes table" so the chosen node cannot be
        #                         revisited later in the same path
        # ---------------------------------------------------------------------

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
            # -----------------------------------------------------------------
            # [Paper §3.2 — Eq. (6)]
            # Quote: "m = MOD(g_j, l_j) + 1"
            # Mapping: gene_value = g_j, l_j = len(available).
            #   The paper uses 1-based indexing (result in [1, l_j]); Python
            #   uses 0-based indexing, so 'available[m]' with m = g_j % l_j
            #   is equivalent to L_j[MOD(g_j, l_j) + 1] in 1-based notation.
            # -----------------------------------------------------------------

            # Eq. (6): m = MOD(g_j, l_j)  → 0-based index into available
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

    # =========================================================================
    # SECTION 2.1 — BBO Migration Rates (Equations 1 and 2)
    # =========================================================================

    def calculate_migration_rates(
        self,
    ) -> Tuple[List[float], List[float]]:
        # ---------------------------------------------------------------------
        # [Paper §2.1 BBO — Eqs. (1) and (2)]
        # Quote: "In BBO, each individual has its own emigration rate λ_s and
        #         immigration rate μ_s, which are functions of the number of
        #         species s; (s = 1, 2 ... i, ... P) in the habitat."
        # Quote: "λ_s = E * s/P"   [Eq. 1]
        # Quote: "μ_s = I * (1 - s/P)"  [Eq. 2]
        # Quote: "where E = max λ_s, I = max μ_s, and P = population size.
        #         Generally, E and I are unit matrix."
        #
        # Mapping:
        #   P                   — population size (self.pop_size)
        #   s                   — species count for a given habitat; computed
        #                         from rank: the best-fitness habitat (rank 0)
        #                         gets s = P (most species, high-quality island)
        #                         and the worst (rank P-1) gets s = 1.
        #   emigration_rates[i] — λ_s for habitat i; a high-quality habitat
        #                         (high s) has HIGH λ, meaning it tends to
        #                         export its features to other habitats.
        #   immigration_rates[i]— μ_s for habitat i; a high-quality habitat
        #                         has LOW μ, meaning it rarely accepts imports
        #                         (it is already a good solution).
        #   E = I = 1           — paper states "generally, E and I are unit
        #                         matrix", so both max rates equal 1.0.
        #
        # Ecological intuition (paper §2.1):
        #   Quote: "High HSI solutions tend to share their features with low
        #           HSI solutions. Poor solutions accept a lot of new features
        #           from good solutions."
        #   A high-HSI (low-cost) habitat has many species → high λ (emigrants)
        #   and low μ (immigrants), meaning it mostly exports good SIVs to
        #   worse habitats rather than receiving features from them.
        # ---------------------------------------------------------------------
        """Return per-habitat emigration (λ) and immigration (μ) rates.

        Habitats are ranked by fitness ascending (best cost = rank 0).
        In the paper's species analogy, a better habitat has more species
        (higher s), giving HIGH λ (exports features) and LOW μ (few imports).

        Eqs. (1) and (2) with E = I = 1:
            λ_s = s / P
            μ_s = 1 − s / P
        where s = P − rank  so that rank 0 (best) → s = P (max species).
        """
        P = self.pop_size

        # argsort returns indices that sort fitness ascending; index 0 = best
        # habitat (lowest cost = highest HSI = most species).
        rank_order = np.argsort(self.fitness)  # ascending: rank_order[0] = best

        emigration_rates = [0.0] * P
        immigration_rates = [0.0] * P

        for rank, habitat_idx in enumerate(rank_order):
            # -----------------------------------------------------------------
            # [Paper §2.1 — Eqs. (1) and (2)]
            # s = species count assigned to this habitat.
            # rank 0 (best)  → s = P → λ = 1.0, μ = 0.0  (pure exporter)
            # rank P-1 (worst) → s = 1 → λ = 1/P ≈ 0, μ ≈ 1.0 (pure importer)
            # -----------------------------------------------------------------
            s = P - rank  # best habitat → s=P, worst → s=1
            emigration_rates[habitat_idx] = s / P  # Eq. (1): λ_s = s/P
            immigration_rates[habitat_idx] = 1.0 - s / P  # Eq. (2): μ_s = 1 - s/P

        return emigration_rates, immigration_rates

    # =========================================================================
    # SECTION 2.2 — PSO Velocity and Position Update (Equation 5)
    # =========================================================================

    def update_velocity_position(self, i: int) -> None:
        # ---------------------------------------------------------------------
        # [Paper §2.2 PSO — Eq. (5)]
        # Quote: "Each particle updates its position and velocity according
        #         to the following equations:
        #           V^{k+1}_i = w*V^k_i + c1*r1*(P_i - X^k_i) + c2*r2*(P_g - X^k_i)
        #           X^{k+1}_i = X^k_i + V^{k+1}_i"
        # Quote: "w is the inertia in range [0.9, 0.4] that increases linearly,
        #         c1 and c2 are two positive constants, usually we choose
        #         c1 and c2 = 2; r1 and r2 are two random functions in the
        #         range [0, 1]."
        #
        # Mapping:
        #   self._w                    — w: inertia weight (decreases linearly
        #                                from w_max=0.9 to w_min=0.4 across
        #                                generations; set by the main loop)
        #   self.c1, self.c2           — cognitive (c1) and social (c2)
        #                                acceleration constants; paper uses 2
        #   r1, r2                     — random scalars in [0, 1]
        #   self.velocities[i][j]      — v^k_{i,j}: current velocity of
        #                                dimension j of particle i (= V^k_i)
        #   self.personal_best[i][j]   — p_{i,j}: personal best position of
        #                                particle i in dimension j (= P_i)
        #   self.population[i][j]      — x^k_{i,j}: current position of
        #                                particle i in dimension j (= X^k_i)
        #   self.global_best[j]        — p_{g,j}: global best position in
        #                                dimension j (= P_g)
        #
        # Context (paper §2.3 BPSO — Step 3.3):
        #   Quote: "If a habitat is NOT selected to be immigrated [by BBO],
        #           update the position of current habitat according to (5)."
        #   This function is therefore called only for habitats that were not
        #   chosen for BBO migration in the current generation.
        # ---------------------------------------------------------------------

        """Update particle i using PSO equations with current inertia weight.

        Fix 3: self._w is set by the main loop to decrease linearly from
        w_max = 0.9 to w_min = 0.4 over max_gen generations.
        """
        r1 = np.random.random()
        r2 = np.random.random()
        n = len(self.velocities[i])

        for j in range(n):
            # -----------------------------------------------------------------
            # [Paper §2.2 — Eq. (5), velocity update]
            # V^{k+1}_{i,j} = w*V^k_{i,j}
            #                + c1*r1*(P_{i,j} - X^k_{i,j})   ← cognitive term
            #                + c2*r2*(P_{g,j} - X^k_{i,j})   ← social term
            # -----------------------------------------------------------------

            self.velocities[i][j] = (
                self._w * self.velocities[i][j]
                + self.c1 * r1 * (self.personal_best[i][j] - self.population[i][j])
                + self.c2 * r2 * (self.global_best[j] - self.population[i][j])
            )

        for j in range(len(self.population[i])):
            # -----------------------------------------------------------------
            # [Paper §2.2 — Eq. (5), position update]
            # X^{k+1}_{i,j} = X^k_{i,j} + V^{k+1}_{i,j}
            # SIV values must remain integer indices into the node list, so we
            # round and clamp to [0, num_nodes − 1].
            # -----------------------------------------------------------------

            self.population[i][j] = int(
                round(self.population[i][j] + self.velocities[i][j])
            )
            # Clamp SIV to valid range [0, num_nodes − 1]
            self.population[i][j] = max(0, min(self.population[i][j], n - 1))

    # =========================================================================
    # SECTION 2.1 — BBO Mutation (Equation 4)
    # =========================================================================

    def mutation(self) -> None:
        # ---------------------------------------------------------------------
        # [Paper §2.1 BBO — Eq. (4) / §2.3 BPSO — Step 3.4]
        # Quote: "The mutation rate m_i is expressed as:
        #           m_i = P_mute * (1 - P_s / P_max)"   [Eq. 4]
        # Quote: "where P_max = arg max P_s; i = 1, ..., P.
        #         P_mute is a parameter deciding mutation rate."
        # Quote: "Mutate the worst half of the population according to (4)."
        #         [§2.3 BPSO Step 3.4]
        #
        # Mapping:
        #   self.p_mutation    — P_mute: base mutation rate; paper uses 0.005
        #   P_s                — probability that habitat i has exactly s
        #                        species (derived from Eq. 3). We approximate
        #                        P_s ≈ s/P (normalised rank probability) since
        #                        Eq. 3's exact differential equation would
        #                        require numerical integration per generation.
        #                        This preserves the key property: worst habitats
        #                        (low s, low P_s) get the highest mutation rate.
        #   P_max              — max P_s = 1.0 (attained by the best habitat,
        #                        which has s = P and P_s = P/P = 1).
        #   m_i                — per-habitat mutation rate; increases as P_s
        #                        decreases, so the worst habitats mutate most.
        #   worst_half         — "the worst half" of the population, sorted by
        #                        descending cost (ascending fitness rank).
        #   siv_idx            — randomly selected SIV position to mutate.
        #   self.population[habitat_idx][siv_idx] = randint(...)
        #                      — "randomly generated SIV (route node) is used
        #                        to replace a SIV in G_i." (paper §3.2 Step 7)
        # ---------------------------------------------------------------------

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
            # -----------------------------------------------------------------
            # [Paper §2.1 — Eq. (4)]
            # global_rank: position of this habitat in the full ranked list
            # s: species count assigned to this rank position
            # P_s: normalised rank probability (≈ P_s from Eq. 3)
            # m_i = P_mute * (1 - P_s / P_max)
            # -----------------------------------------------------------------
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

    # =========================================================================
    # SECTION 3.2 — Step 4: Fitness Function (Equation 7)
    # =========================================================================

    def calculate_fitness(
        self,
        path: List[int],
        distance_matrix: Dict[int, Dict[int, float]],
        end_node: int,
    ) -> float:
        # ---------------------------------------------------------------------
        # [Paper §3.2 — Step 4 / Eq. (7)]
        # Quote: "After decoding the habitat, the cost of every branch of AVBN
        #         is calculated by adding up at the grids on path between every
        #         two nodes based on the obtained route. The sum of the distance
        #         between two nodes of a route and the punishing cost is the
        #         total cost of a habitat. That is,
        #           C(G_i) = C(S_i) = Σ_{i=1}^{N} D(s_i, s_{i+1}) + k*α"
        #         [Eq. 7]
        # Quote: "where k = 1, if the habitat is useless, else k = 0;
        #         α is the punishment, N is the path node number of S_i."
        #
        # Mapping:
        #   path                — S_i: the decoded sequence of route nodes for
        #                         habitat G_i (output of decode_path)
        #   distance_matrix[n1][n2]
        #                       — D(s_i, s_{i+1}): the number of grid cells
        #                         along the Voronoi boundary branch connecting
        #                         node n1 to node n2; computed by BFS in
        #                         AVBN._build_connections()
        #   total_cost          — Σ D(s_i, s_{i+1}): running sum of branch
        #                         lengths over the entire decoded path
        #   alpha = 1000        — α: the punishment cost applied when the
        #                         path fails to reach the goal node
        #   k_penalty           — k: 0 if path[-1] == end_node (successful
        #                         path), 1 otherwise (useless habitat)
        #   return 1e6          — sentinel for truly invalid paths (fewer
        #                         than 2 nodes, or a broken graph edge);
        #                         much larger than any real cost so BPSO
        #                         never selects these as personal/global best
        # ---------------------------------------------------------------------
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
                return 1e6  # disconnected → invalid path
            total_cost += seg

        alpha = 1000
        k_penalty = 0 if path[-1] == end_node else 1
        return total_cost + k_penalty * alpha

    # =========================================================================
    # SECTION 2.3 — Full BPSO Optimisation Loop
    # =========================================================================

    def optimize(
        self,
        nodes: List[Tuple[int, int]],
        node_connections: Dict[int, List[int]],
        distance_matrix: Dict[int, Dict[int, float]],
        start_node: int,
        end_node: int,
    ) -> Tuple[List[int], float, dict]:
        # ---------------------------------------------------------------------
        # [Paper §2.3 — BPSO procedure overview]
        # Quote: "Step 1: Initialize parameters: P, G, P_mute, w, c1, c2.
        #         Step 2: Evaluate the fitness for each individual in P.
        #         Step 3: While the termination criteria is not met do
        #           Step 3.1: Record the previous best position of each habitat
        #                     P_i and their neighbourhood best position P_g.
        #           Step 3.2: For each habitat map the HSI to number of species
        #                     s, λ_i and μ_i, probabilistically choose the
        #                     immigration island based on the immigration rates.
        #           Step 3.3: If a habitat is selected to be immigrated,
        #                     migrate randomly selected SIVs based on the
        #                     selected island; else update the position of
        #                     current habitat according to (5).
        #           Step 3.4: Mutate the worst half of the population.
        #           Step 3.5: Evaluate the fitness for each individual in P
        #                     and sort the population from best to worst, save
        #                     the best two habitats.
        #         Step 4: End while"
        # Mapping: The structure of the for-loop below mirrors Steps 3.1–3.5
        #   exactly, with the variable names matching those in the paper.
        # ---------------------------------------------------------------------

        """Run BPSO; return (best_path_node_indices, best_cost, history)."""
        num_nodes = len(nodes)
        self.initialize_population(num_nodes)

        history: Dict[str, List] = {
            "best_fitness": [],
            "mean_fitness": [],
            "diversity": [],  # avg pairwise Hamming distance between habitats
            "node_visit_counts": np.zeros(num_nodes, dtype=int),  # heatmap data
        }

        for generation in range(self.max_gen):
            # -----------------------------------------------------------------
            # [Paper §2.2 PSO — Eq. (5) inertia schedule]
            # Quote: "w is the inertia in range [0.9, 0.4] that increases
            #         linearly."
            # Mapping: w decreases from w_max=0.9 to w_min=0.4 linearly over
            #   max_gen generations. A high w at the start encourages broad
            #   exploration; a low w at the end encourages fine exploitation.
            # -----------------------------------------------------------------

            # ---- Fix 3: linearly decreasing inertia weight ---------------
            self._w = self.w_max - (self.w_max - self.w_min) * generation / max(
                self.max_gen - 1, 1
            )

            # -----------------------------------------------------------------
            # [Paper §2.3 BPSO — Step 3.1 / §3.2 Step 4]
            # Quote: "Step 3.1: Record the previous best position of each
            #         habitat P_i and their neighbourhood best position P_g."
            # Mapping:
            #   self.fitness[i]           — C(G_i) from Eq. (7) for habitat i
            #   self.personal_best[i]     — P_i: best encoding seen so far
            #   self.global_best          — P_g: best encoding across all
            #                              habitats and all generations
            # -----------------------------------------------------------------
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

            # -----------------------------------------------------------------
            # [Paper §2.3 BPSO — Step 3.5]
            # Quote: "Evaluate the fitness for each individual in P and sort
            #         the population from best to worst, save the best two
            #         habitats."
            # Mapping: We save the best two habitats and their fitnesses
            #   *before* migration and mutation so that they can be restored
            #   afterwards (elite preservation). This prevents good solutions
            #   from being overwritten during the stochastic update steps.
            # -----------------------------------------------------------------

            # ---- Fix 4: save elites before modification ------------------
            rank_order = np.argsort(self.fitness)
            elite_indices = rank_order[:2]
            elite_habitats = [copy.deepcopy(self.population[i]) for i in elite_indices]
            elite_fitnesses = [self.fitness[i] for i in elite_indices]

            # -----------------------------------------------------------------
            # [Paper §2.3 BPSO — Step 3.2 / §2.1 BBO migration rates]
            # Quote: "For each habitat map the HSI to number of species s,
            #         λ_i and μ_i, probabilistically choose the immigration
            #         island based on the immigration rates."
            # Mapping: calculate_migration_rates() returns emigration_rates
            #   (λ) and immigration_rates (μ) per habitat, computed from
            #   Eqs. (1) and (2) using the current fitness ranking.
            # -----------------------------------------------------------------

            # ---- Step 3.2: compute migration rates -----------------------
            emigration_rates, immigration_rates = self.calculate_migration_rates()

            # -----------------------------------------------------------------
            # [Paper §2.3 BPSO — Steps 3.2 and 3.3 / §2.1 BBO migration]
            # Quote (migration pseudocode §2.1):
            #   "Step 1: For i=1 to P
            #    Step 2: Select H_i with λ_i
            #    Step 3: If H_i is selected
            #    Step 4:   For j=1 to P
            #    Step 5:   Select H_j with probability μ_i [sic; μ_j]
            #    Step 6:   If H_j is selected
            #    Step 7:   Randomly select an SIV σ from H_j
            #    Step 8:   Replace a random SIV in H_i with σ"
            # Quote (BPSO distinction §2.3):
            #   "If a habitat is selected to be immigrated migrate randomly
            #    selected SIVs based on the selected island
            #    else update the position of current habitat according to (5)"
            #
            # Mapping:
            #   emigration_rates[i] — λ_i: probability that habitat H_i is
            #                          selected for immigration (Step 2).
            #                          A high-quality habitat (high s) has a
            #                          high λ and is therefore frequently
            #                          chosen as a *recipient* of new SIVs.
            #   immigration_rates   — μ_j: used to weight the selection of
            #                          the *source* habitat H_j (Step 5).
            #                          A high-quality source (high μ … wait:
            #                          high-quality has LOW μ; the paper's
            #                          source selection is proportional to μ_j
            #                          so that poorer habitats donate more).
            #   siv_idx             — the randomly chosen SIV index (σ
            #                          position) in H_j (Step 7).
            #   new_population[i][siv_idx] = self.population[j][siv_idx]
            #                      — "Replace a random SIV in H_i with σ"
            #                          (Step 8).
            # -----------------------------------------------------------------
            # ---- Steps 3.2–3.3: migration + PSO fallback -----------------
            # choose source Hj weighted by immigration rates μ_j.
            new_population = copy.deepcopy(self.population)

            for i in range(self.pop_size):
                if np.random.random() < emigration_rates[i]:
                    # Habitat i is selected → receive an SIV from a source
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
                    # ---------------------------------------------------------
                    # [Paper §2.3 BPSO — Step 3.3, PSO fallback]
                    # Quote: "In the original BBO, if there is no habitat
                    #         selected for immigration, the selected habitat
                    #         for emigration is not changed. In this case, it
                    #         has no benefit for increasing the diversity of
                    #         population. Inspired by the concept of PSO, we
                    #         use the particle position strategy to modify the
                    #         habitat which is not selected to be immigrated."
                    # Mapping: Habitats that escape the migration step are
                    #   updated using PSO Eq. (5) via update_velocity_position,
                    #   preserving population diversity — the core innovation
                    #   of BPSO over plain BBO.
                    # ---------------------------------------------------------

                    # Not selected for immigration → PSO position update
                    if self.global_best:
                        self.population[i] = new_population[i]  # sync first
                        self.update_velocity_position(i)
                        new_population[i] = copy.deepcopy(self.population[i])

            self.population = new_population

            # -----------------------------------------------------------------
            # [Paper §2.3 BPSO — Step 3.4 / §2.1 BBO mutation]
            # Quote: "Mutate the worst half of the population according to (4)."
            # Mapping: self.mutation() applies Eq. (4) only to the lower half
            #   of the population ranked by fitness (ascending order), so the
            #   worst-performing habitats receive the highest mutation rates.
            # -----------------------------------------------------------------
            # ---- Step 3.4: mutate worst half -----------------------------
            self.mutation()

            # -----------------------------------------------------------------
            # [Paper §2.3 BPSO — Step 3.5, elite preservation]
            # Quote: "Evaluate the fitness for each individual in P and sort
            #         the population from best to worst, save the best two
            #         habitats."
            # Mapping: We restore the two saved elite habitats into the two
            #   worst positions of the post-mutation population, ensuring the
            #   best solutions found so far can never be discarded.
            # -----------------------------------------------------------------
            # ---- Fix 4: restore elites (step 3.5) -----------------------
            # Replace the two currently-worst habitats with the saved elites
            post_rank = np.argsort(self.fitness)
            worst_two = post_rank[-2:]
            for slot, (hab, fit) in enumerate(zip(elite_habitats, elite_fitnesses)):
                replace_idx = worst_two[slot]
                self.population[replace_idx] = hab
                self.fitness[replace_idx] = fit

            # -----------------------------------------------------------------
            # Diagnostic: node visit counts for search-space heatmap.
            # Not from the paper — records how often each AVBN node appears
            # in the population's decoded paths each generation, used by the
            # Plot 4 heatmap in analysis.py to visualise which parts of the
            # network BPSO explores.
            # -----------------------------------------------------------------
            # ---- Node visit counts (for heatmap) ------------------------
            # Record which nodes appear in every individual's decoded path
            for individual in self.population:
                path = self.decode_path(
                    individual, node_connections, start_node, end_node
                )
                for node_idx in path:
                    if 0 <= node_idx < num_nodes:
                        history["node_visit_counts"][node_idx] += 1

            # -----------------------------------------------------------------
            # Diagnostic: population diversity.
            # Not from the paper — measures the average pairwise Hamming
            # distance between all habitat SIV vectors (normalised by genome
            # length). Used by Plot 3 in analysis.py to verify that the PSO
            # fallback (§2.3 Step 3.3) genuinely maintains diversity compared
            # to plain BBO.
            # -----------------------------------------------------------------
            # ---- Population diversity ------------------------------------
            # Average pairwise Hamming distance between all habitat SIV vectors
            # (treating integer SIVs as a discrete genotype)
            pop_arr = np.array(self.population, dtype=float)
            n = len(pop_arr)
            if n > 1:
                # Vectorised: for each pair compute element-wise inequality
                diffs = 0.0
                for a in range(n):
                    diffs += np.sum(pop_arr[a] != pop_arr, axis=1).sum()
                # Each pair counted twice; normalise by genome length
                genome_len = pop_arr.shape[1] if pop_arr.ndim == 2 else 1
                avg_hamming = diffs / (n * (n - 1)) / genome_len
            else:
                avg_hamming = 0.0
            history["diversity"].append(float(avg_hamming))

            # -----------------------------------------------------------------
            # Bookkeeping: record convergence data for Plot 0.
            # best_fitness[g] = global best C(G_i) at the end of generation g.
            # mean_fitness[g] = average C(G_i) across the population.
            # -----------------------------------------------------------------
            # ---- Bookkeeping ---------------------------------------------
            history["best_fitness"].append(self.global_best_fitness)
            history["mean_fitness"].append(float(np.mean(self.fitness)))

            if generation % 10 == 0:
                print(
                    f"  Gen {generation:3d} | w={self._w:.3f} | "
                    f"Best = {self.global_best_fitness:.2f} | "
                    f"Mean = {np.mean(self.fitness):.2f}"
                )

        # ---------------------------------------------------------------------
        # [Paper §2.3 BPSO — Step 4: termination]
        # Quote: "The cycle is ended when there is no best habitat emerging
        #         after some generations."
        # Mapping: We run for a fixed max_gen (= G in the paper, set to 100
        #   in §4.2) and return the globally best habitat decoded into a path.
        # ---------------------------------------------------------------------
        best_path = self.decode_path(
            self.global_best, node_connections, start_node, end_node
        )
        return best_path, self.global_best_fitness, history


# =============================================================================
# Module-level helper
# =============================================================================


def _remove_from_all(connections: Dict[int, List[int]], node: int) -> None:
    # -------------------------------------------------------------------------
    # [Paper §3.2 — Step 3 (decoding), visited-node deletion]
    # Quote: "When a route node is selected to one of the path nodes, it will
    #         be deleted from all correlated nodes table for avoiding being
    #         reselected. This operation continues till we get a solution S."
    # Mapping:
    #   connections — local_conn in decode_path: a per-call mutable copy of
    #                 the correlated nodes table (L_j for every j).
    #   node        — the route node just added to the path; it must be
    #                 removed from *every* adjacency list, not just from the
    #                 list of the current node, because any future step along
    #                 any branch could otherwise re-select it.
    # -------------------------------------------------------------------------
    """Remove *node* from every adjacency list in *connections*."""
    for adj in connections.values():
        try:
            adj.remove(node)
        except ValueError:
            pass
