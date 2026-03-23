# Robot Path Planning via BPSO-AVBN

This project implements a global path planning system for mobile robots in static environments by combining **Biogeography-based Optimization (BBO)**, **Particle Swarm Optimization (PSO)**, and an **Approximate Voronoi Boundary Network (AVBN)**.

## Overview
The core challenge in Robot Path Planning (RPP) is finding an optimal collision-free path from a start point to a goal in a complex environment. This implementation utilizes a hybrid algorithm (BPSO) to leverage the information-sharing capabilities of BBO and the diversity-increasing position updates of PSO.

### Key Components
* **AVBN (Approximate Voronoi Boundary Network):** Simplifies the environment by enlarging obstacle clusters and extracting a non-smooth path network based on Voronoi boundaries.
* **BBO (Biogeography-based Optimization):** Uses migration operators to share features (Suitability Index Variables or SIVs) between candidate solutions based on their Habitat Suitability Index (HSI).
* **PSO (Particle Swarm Optimization):** Enhances population diversity by applying a position updating strategy to habitats that are not selected for migration.

## Project Structure
```text
.
├── docs/
│   └── Research of biogeography particle swarm optimization...pdf
├── src/
│   ├── avbn.py                # Environment modeling & Voronoi logic
│   ├── bpso.py                # Hybrid BPSO algorithm implementation
│   ├── main.py                # Experiment runner & visualization
│   └── robot_path_planner.py  # High-level API for the planner
├── output/                    # Generated plots and results
├── README.md                  # Project documentation
└── requirements.txt           # Project dependencies
```

## Algorithm Logic
1.  **Environment Modeling**: The workspace is divided into a $400 \times 400$ grid where obstacles are enlarged to ensure a safety margin for the robot.
2.  **Network Construction**: The AVBN identifies nodes at the intersections of Voronoi boundaries, transforming the continuous planning problem into a discrete network search.
3.  **Path Optimization**:
    * **Encoding**: Each individual (habitat) represents a series of route nodes.
    * **Migration**: High-fitness solutions share features with low-fitness solutions.
    * **PSO Update**: Habitats not modified by migration are updated using PSO velocity and position equations to prevent local optima.
    * **Mutation**: The worst half of the population undergoes random mutation to maintain diversity.

## Installation & Usage
1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd rpp
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Simulation**:
    ```bash
    python src/main.py
    ```

## Results
Based on the research findings, the BPSO-AVBN approach demonstrates:
* **Efficiency**: Higher success rates in finding the global optimum compared to standard GA, PSO, or BBO.
* **Convergence**: Faster convergence speeds due to the combined strengths of migration and swarming behavior.
