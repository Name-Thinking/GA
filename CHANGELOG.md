# 📝 Changelog - GeneTeam

All notable changes to the Genetic Algorithm sports division system.

## [1.0.0] - 2026-11-05
### Added
- **Core Genetic Algorithm Engine (`src/ga.py`)**:
  - Binary Chromosome representation for 2-team partitioning (Red vs Green).
  - Balanced Initialization ensuring 50:50 ratio.
  - Swap Mutation operator preserving size equality.
  - Two-point crossover with size repair mechanism.
  - Tournament Selection ($k=3$) and Elitism ($e=2$).
- **Multi-Objective Fitness Function (`src/fitness.py`)**:
  - Sports Fairness & Engagement score ($\Delta\mu < 0.1$, balanced $\sigma$).
  - Cross-Department Harmonization score (Department parity & Shannon diversity entropy).
  - Size constraint penalty.
- **Visualization Suite (`src/visualizer.py`)**:
  - Fitness convergence curve (`outputs/fitness_evolution.png`).
  - Department distribution grouped bar chart (`outputs/department_distribution.png`).
  - Sports skill boxplot & summary metrics (`outputs/sports_balance.png`).
  - Self-contained interactive HTML dashboard (`outputs/report.html`).
- **Comprehensive Unit Tests (`tests/test_ga.py`)**:
  - 6/6 test cases covering data loading, chromosome balance, repair, mutation, and convergence.

