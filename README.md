# After the Ice Storm
## Optimization of Emergency Communication Network Deployment

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-Pytest-green)](https://pytest.org/)

**After the Ice Storm** is a computational optimization project focused on deploying portable radio communication stations in the fictional Greyhaven region following a severe ice storm.

The project implements and evaluates continuous, mixed-variable, and multi-objective optimization methods to identify effective emergency communication network configurations under different operational scenarios.

The project is organized into three assignments (A, B, and C). The central experiment orchestration interface is provided by `frostline_runner.py`.

---

## 1. Problem Overview

The problem considers an emergency communication network in a square geographical region:

- Service area: \([0,10]\times[0,10]\).
- 18 fixed demand points.
- Four portable radio stations.
- Fixed deployment depot at \((5,0)\).

The objective is to determine station configurations that provide effective communication service while accounting for deployment constraints, interference, operational cost, and energy consumption.

Three operational scenarios are considered:

1. **Initial Response:** Initial emergency response conditions.
2. **Evening Pressure:** Increased demand at selected locations during the evening.
3. **Northern Outage:** Increased operational pressure in the northern part of the region.

The mathematical model includes station placement, transmission power, antenna height, channel assignment, activation status, signal strength, interference, and service quality.

The service-quality measure is a dimensionless optimization index. It should not be interpreted as a measured probability of successful communication or as actual network throughput.

---

## 2. Optimization Assignments

### Assignment A — Continuous Optimization

Assignment A considers a continuous deployment formulation.

- Decision variables: station coordinates and transmission powers.
- Antenna height: fixed at 4.
- Channel: fixed at 1.
- All stations: active.
- Objective: minimize the negative weighted service-quality measure \(Q\).

The following algorithms are implemented:

| Algorithm | Approach |
|---|---|
| Differential Evolution (DE) | Population-based evolutionary optimization |
| Particle Swarm Optimization (PSO) | Swarm-based optimization |
| Covariance Matrix Adaptation Evolution Strategy (CMA-ES) | Adaptive stochastic search |

### Assignment B — Mixed-Variable Optimization

Assignment B extends the decision space to include continuous and discrete deployment decisions.

Each station is characterized by:

- \(x,y\): geographical coordinates.
- \(p\): transmission power.
- \(h\): antenna height.
- \(c\): communication channel.
- \(z\): activation status.

The formulation contains 24 decision variables across four stations.

The objective is to minimize \(Q\).

| Algorithm | Approach |
|---|---|
| MVGA | Mixed-Variable Genetic Algorithm |
| MVDE | Mixed-Variable Differential Evolution |
| MVSA | Mixed-Variable Simulated Annealing |

### Assignment C — Multi-Objective Optimization

Assignment C uses the same mixed-variable decision space as Assignment B but considers three objectives simultaneously:

- \(Q\): negative weighted service quality.
- \(C\): deployment cost.
- \(E\): energy consumption.

All three objectives are minimized.

| Algorithm | Approach |
|---|---|
| MVNSGA-II | Mixed-Variable Non-dominated Sorting Genetic Algorithm II |
| MOEA/D | Multi-Objective Evolutionary Algorithm Based on Decomposition |
| SPEA2 | Strength Pareto Evolutionary Algorithm 2 |

The resulting non-dominated solutions represent alternative trade-offs between service quality, deployment cost, and energy consumption.

Objective normalization is implemented to support consistent multi-objective comparisons and distance-based calculations.

---

## 3. Experimental Design

The experimental framework supports reproducible comparisons across algorithms and operational scenarios.

The principal experimental settings are:

| Setting | Value |
|---|---:|
| Independent runs per algorithm and scenario | 31 |
| Maximum deployment evaluations per run | 10,000 |
| Operational scenarios | 3 |
| Number of stations | 4 |
| Number of demand points | 18 |

Each independent run uses its own random seed.

The evaluation budget applies to deployment evaluations. Algorithm initialization and optimization procedures are subject to the same evaluation-accounting requirements.

Algorithm performance is assessed using descriptive and statistical analyses. For multi-objective optimization, Pareto dominance and hypervolume are used to examine the quality of generated solution sets.

Where applicable, statistical comparisons use paired tests and multiple-comparison corrections.

The 10,000-evaluation setting is a maximum budget; actual evaluation counts may differ between algorithms due to their stopping criteria and implementation.

---

## 4. Repository Structure

```text
After_the_ice_storm/
├── frostline_runner.py
├── pyproject.toml
├── requirements.txt
├── README.md
├── src/
│   └── ice_storm/
│       ├── assignments/
│       │   ├── assignment_a/
│       │   ├── assignment_b/
│       │   └── assignment_c/
│       ├── core/
│       └── optimization/
│           └── algorithms/
│               ├── continuous/
│               ├── mixed_variable/
│               └── multiobjective/
├── tests/
│   ├── test_assignments/
│   ├── test_core/
│   └── additional test modules
├── scripts/
│   └── select_assignment_c_operational_solutions.py
└── results/
    ├── final/
    │   ├── assignment_a/
    │   ├── assignment_b/
    │   └── assignment_c/
    └── selected Assignment B tuning CSV files
```

### Main Components

- **core:** Mathematical models, deployment representation, scenarios, signal propagation, interference, service evaluation, and objective calculations.
- **assignments:** Problem formulations, experiment execution, and analysis modules for Assignments A, B, and C.
- **optimization:** Continuous, mixed-variable, and multi-objective optimization algorithms.
- **tests:** Unit and integration tests for mathematical models, optimization methods, objective scaling, and experiment-runner behavior.
- **scripts:** Supporting analysis and operational solution-selection utilities.
- **results/final:** Selected experimental results, Pareto populations, statistical summaries, and analysis visualizations.
- **frostline_runner.py:** Central command-line interface for running the assignments.

---

## 5. Installation and Setup

### Requirements

- Python 3.13 or later.
- NumPy
- SciPy
- Pandas
- Matplotlib
- CMA-ES (`cma`)
- Pytest (for development and testing)

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### Install dependencies

Install the project and its development dependencies:

```bash
pip install -e ".[dev]"
```

Alternatively, install the listed project dependencies:

```bash
pip install -r requirements.txt
```

---

## 6. Running Experiments

All three assignments can be executed through the central experiment runner.

### Display available options

```bash
python frostline_runner.py --help
```

### Run individual assignments

```bash
python frostline_runner.py --assignment A
```

```bash
python frostline_runner.py --assignment B
```

```bash
python frostline_runner.py --assignment C
```

### Run all assignments

```bash
python frostline_runner.py --assignment all
```

By default, the runner uses:

- 31 independent runs.
- A maximum of 10,000 evaluations per run.
- The default algorithm configurations defined in the corresponding experiment modules.

### Customize experimental settings

For a smaller test execution:

```bash
python frostline_runner.py --assignment A --runs 1 --max-evaluations 1000
```

The same options can be used for Assignments B and C.

### Dry run

To inspect the execution plan without starting experiments:

```bash
python frostline_runner.py --assignment all --dry-run
```

The dry-run option displays the selected assignments, run count, evaluation budget, and output directory without executing optimization algorithms.

### Output location

By default, the central runner writes its outputs to:

```text
results/frostline_runner/
```

A different output directory can be specified using `--output-dir`.

**Note:** Full experimental runs can be computationally intensive. The default configuration is intended for the complete experimental design rather than a quick smoke test.

---

## 7. Testing and Validation

The project uses `pytest` for automated testing.

Run the complete test suite:

```bash
python -m pytest -q
```

Run tests with detailed output:

```bash
python -m pytest -v
```

The test suite covers:

- Deployment representation and constraints.
- Signal and interference calculations.
- Service-quality evaluation.
- Objective calculations.
- Assignment-specific problem formulations.
- Continuous and mixed-variable optimization algorithms.
- Multi-objective optimization utilities.
- Objective normalization.
- Central experiment-runner behavior.

Tests are intended to verify mathematical consistency, implementation correctness, and expected algorithm behavior independently of full experimental runs.

---

## 8. Results and Reproducibility

Selected experimental results are organized under `results/final/`.

The published results include:

- Assignment A run-level results and statistical comparisons.
- Assignment B run-level results, objective summaries, convergence analysis, and selected parameter-tuning records.
- Assignment C run-level results, Pareto populations, objective analyses, and operational solution-selection outputs.

The experimental framework records run-level information to support comparisons across algorithms and scenarios.

Reproducibility is supported through:

- Explicit random-seed management.
- Configurable run counts and evaluation budgets.
- Consistent scenario definitions.
- Automated tests.
- Separate experiment modules for each assignment.
- Centralized experiment orchestration.

Previously generated results should be preserved when conducting additional experiments. New executions can use a separate output directory through the `--output-dir` option.

The Assignment C operational solution-selection utility is available at:

```text
scripts/select_assignment_c_operational_solutions.py
```

It reads the selected Pareto populations, combines and filters non-dominated solutions, applies the minimum service-quality requirement, and selects an operational candidate based on normalized cost-energy distance.

---

## 9. Modeling Assumptions and Limitations

The project uses a simplified mathematical representation of emergency radio communication deployment.

The service model is intended for comparative optimization rather than direct prediction of real-world network performance.

In particular:

- The geographical region and demand-point locations are fixed.
- The propagation and interference models use simplified assumptions.
- The service-quality index is dimensionless.
- Detailed traffic-capacity constraints and time-varying propagation effects are not explicitly modeled.
- Real-world terrain, equipment characteristics, and operational uncertainties may require additional modeling.

Consequently, the resulting deployments should be interpreted as optimization solutions under the stated assumptions, not as field-validated communication network designs.

---

## 10. Project Information

**Project:** After the Ice Storm — Optimization of Emergency Communication Network Deployment

**Repository:**
https://github.com/NesaSadeghi2018/After_the_ice_storm

This repository provides the computational implementation, experimental framework, automated tests, and selected results for examining continuous, mixed-variable, and multi-objective optimization approaches to emergency communication network deployment.