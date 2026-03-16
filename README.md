# $\lambda$-Precision Unit Disk Graph (UDG) Generator

A Python toolkit for generating $\lambda$-precision Unit Disk Graphs (UDGs), primarily designed for modelling
large-scale static wireless sensor networks (WSNs). It features a suite of solver-agnostic MILPs. Whether you are minimising network variance or balancing complex resource distributions, it provides the algorithms needed to perfectly allocate resources across distributed networks.

## Overview

This toolkit provides tools to:

- Generate $\lambda$-precision UDGs with specific properties for network simulations.
- Evaluate partitioning, assignment, and distribution schemes using mixed-integer linear or non-linear programming (
  MILP/MINLP).
- Export results (e.g., tables) in formats like LaTeX using `tabulate`.

### Research Contributions

Earlier versions of this toolkit contributed to these publications:

- **Determining Distributions of Security Means for WSNs Based on the Model of a Neighbourhood Watch
  ** [@forster2024determining].
- **Security Mean Distribution in WSNs for Cooperative Schemes** [@forster2024security].
- **Topology-and Resource-Based Distribution Scheme for Collaborative Security-Focused Design Space Exploration in
  Large-Scale Static WSNs** [@forster2024topology].

---

## Features

- **Graph Generation:**
    - Create $\lambda$-precision geometric graphs with tunable density and connectivity.
    - Generate graphs with guaranteed properties such as connectivity or degree constraints.

- **Optimisation Models:**
    - Solve partitioning, distribution, and resource assignments through MILP/MINLP formulations.
    - Includes various optimisation methods (e.g., minimising variance, spread, or maximising coverage).

- **Visualisation:**
    - Graphically illustrate generated graphs and optimisation results.

- **Export Capabilities:**
    - Generate tabular outputs for visualisation or report writing.
    - Plot results with matplotlib
    - Direct LaTeX export for processed data tables.

---

## Installation

### Prerequisites

Ensure you have the following installed and set up on your system:

- **Python:** Version 3.10 or higher.
- **Poetry:** A dependency management tool for Python.

### Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/bfoerster/lambda_precision_udg_generator.git
   cd lambda_precision_udg_generator
   ```

2. **Install Dependencies Using Poetry:**

   ```bash
   poetry install
   ```

3. **Activate the Virtual Environment:**

   ```bash
   poetry shell
   ```

---

## Usage

### Generating Graphs

The toolkit allows you to generate $\lambda$-precision UDGs based on your specifications:

- **Points Generation:** Use `RandomPointsGenerator` to define the number of points and minimum distances.
- **Graph Properties:** Define connectivity radius, degree constraints, etc.

Example (Python):

```python
from lambdaprecisionudggenerator.graph_generator import LambdaPrecisionUDGGenerator
from lambdaprecisionudggenerator.graph_generator.points.generator import RandomPointsGenerator

# Create a random points generator:
random_points_gen = RandomPointsGenerator(point_number=300, min_dist=0.037)

# Generate a graph using the LambdaPrecisionUDGGenerator:
graph_generator = LambdaPrecisionUDGGenerator(random_points_gen, radius=0.083)
generated_graph = graph_generator.generate_graph(connected=True)

print(f"Graph has {len(generated_graph.nodes)} nodes and {len(generated_graph.edges)} edges.")
```

### Optimisation Examples

Refer to the `src/partitioning/` module to explore MILP/MINLP optimisation models. The library offers tools to minimise
variance, spread, or other measures.

### Testing the Toolkit

Utilise the integrated test suite based on `pytest` to verify the functionality.

```bash
pytest tests/
```

---

## Documentation

This project uses **MKDocs** to generate and host documentation.

### Building Documentation

To build the documentation locally:

1. **Install Prerequisites:**
   Ensure `pandoc` is installed on your system.

2. **Build the Documentation:**

   ```bash
   poetry run mkdocs build --clean
   ```

3. **Serve the Documentation Locally:**

   ```bash
   poetry run mkdocs serve
   ```

   Open the provided local URL in your browser to view the generated docs.

---

## Project Structure

```
LambdaPrecisionUDGGenerator/
├── src/
│   ├── graph_generator/    # Logic for generating graphs and points
│   ├── partitioning/       # Optimisation models for partitioning
│   ├── utils/              # Utilities like logging configuration
├── docs/                   # Documentation files
├── README.md               # Project overview and setup guide
└── pyproject.toml          # Poetry configuration
```

Highlights of key directories:

- **`src/graph_generator/`:** Generates points and graphs, ensuring desired $\lambda$ precision and density properties.
- **`src/partitioning/`:** Implements optimisation models for resource and partitioning schemes.
- **`tests/`:** Contains `pytest`-based test cases to validate the toolkit.

---

## Contributing

Contributions to the project are welcomed! Please follow these steps:

1. Fork the repository and create a branch for your feature or hotfix.
2. Commit your changes and open a pull request.
3. Ensure your code passes all tests and adheres to quality guidelines.

### Development Environment

After cloning the repository, set up the development environment by installing the `dev` dependencies:

```bash
poetry install --with dev
```

---

## License

This project is licensed under the European Union Public License (EUPL) v1.2.

The EUPL is a free software license that allows you to use, modify, and distribute the software under certain
conditions. By using this project, you agree to comply with the terms of the EUPL v1.2.

For more details on the license, please refer to the full text of the license available at EUPL v1.2 License.
