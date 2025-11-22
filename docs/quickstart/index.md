# $\lambda$ - precision UDG Generator as Model for Large Scale Static Wireless Sensor Networks

A Python toolkit to generate $\lambda$-precision UDGs with specific properties as model for large-scale static wireless
sensor networks (WSNs) and to evaluate the computability of mixed integer (non-)linear programs for mean(s) to node
assignments (in form of partitionings or distributions).

Older iterations of this toolkit have been used to create the following research publications:

- Determining Distributions of Security Means for WSNs based on the Model of a Neighbourhood
  Watch [@forster2024determining]
- Security Mean Distribution in WSNs for Cooperative Schemes [@forster2024security]
- Topology-and resource-based distribution scheme for collaborative security-focused design space exploration in
  large-scale static WSNs [@forster2024topology]

## Commands

To setup all necessary dependencies run:

* `poetry install`

To build the documentation run:

* `poetry run mkdocs build --clean`

To execute test cases run:

* `poetry run pytest -s --full-trace tests/partitioning.py::test_partitioning_spread_resource_based`
