# $\lambda$-Precision Unit Disk Generator (UDG)

A Python toolkit to generate $\lambda$-precision UDGs with specific properties as model for large-scale static wireless
sensor networks (WSNs) and to evaluate the computability of mixed integer (non-)linear programs on those.

Older iterations of this toolkit have been used to create the following research publications:

- Determining Distributions of Security Means for WSNs based on the Model of a Neighbourhood
  Watch [@forster2024determining]
- Security Mean Distribution in WSNs for Cooperative Schemes [@forster2024security]
- Topology-and resource-based distribution scheme for collaborative security-focused design space exploration in
  large-scale static WSNs [@forster2024topology]

## Features

- Create $\lambda$-precision UDGs as models for large-scale static WSNs with specific properties
- Evaluate specific partitioning/assignment/distribution schemes in form of mixed integer linear/non-linear programs
- Generate rectangular tables and export them as LaTeX via `tabulate`

## Installation

### Prerequisites

- Python 3.10+
- Poetry (dependency management)

### Setup

1. **Clone the repository**:

    ```bash
    git clone bfoerster/lambda_precision_udg_generator.git
    cd lambda_precision_udg_generator
    ```

2. **Install dependencies**:
    ```bash
    poetry install
    ```


3. **Activate the virtual environment**:
    ```bash
    poetry shell
    ```

## Generate Documentation

The documentation is generated using MKDocs.

### Prerequisites

- Pandoc

### Setup

1. **Build the documentation**:
   ```bash
   poetry run mkdocs build --clean
   ```

2. **Run a server to automatically update/rebuild to the documentation**
   ```bash
   poetry run mkdocs serve
   ```

## Usage

The usage examples for the toolkit can be derived from test cases in `src/tests`.

## Project Structure

- `src/graph_generator/points` creates a random uniform distribution of points with $\lambda$-precision, hence between
  each pair of points is a minimal distance of $\lambda\in\mathbb{R}^+$
- `src/graph_generator/graphs` utilises NetworkX library to generate geometric graphs using the generated points and
  ensures additional properties like the graph being connected and providing a specific average node degree
- `src/graph_generator/seeds` allows to determines a number of parameter combinations to generate large numbers of
  graphs with
  specific properties stored in `seed.GeneratorSeed`. Multiple seeds are stored in `seed.database.GeneratorSeedDB` which
  provides methods to evaluate and depict their properties as well as serialise and deserialise seeds.
- `src/partitioning` uses pyomo to compute mixed integer (non-)linear programs and to evaluate them. Details are given
  the reference publications. The used default solver is `gurobi`.
- `src/utils` contains a global setup for a logger.
- `tests/` – pytest suite contains multiple test and provide examples on how to utilise the toolkit for your own
  purposes

## License