"""
Configuration and Hyperparameters for Genetic Algorithm Team Division
"""
from dataclasses import dataclass

@dataclass
class GAConfig:
    # Genetic Algorithm Parameters
    population_size: int = 100
    generations: int = 150
    crossover_rate: float = 0.85
    mutation_rate: float = 0.15
    tournament_size: int = 3
    elitism_count: int = 2
    random_seed: int = 42
    
    # Fitness Weights
    weight_sports: float = 0.50      # Weight for sports fairness & engagement
    weight_harmonize: float = 0.50   # Weight for department mixing & diversity
    penalty_size_weight: float = 5.0 # Penalty per unit of team size imbalance
    
    # Sub-weights for Harmonize
    harmonize_parity_weight: float = 0.50     # Balance of each dept across teams
    harmonize_entropy_weight: float = 0.50    # Dept diversity within each team
    
    # Sub-weights for Sports
    sports_mean_weight: float = 0.70          # Difference in average skill
    sports_std_weight: float = 0.30           # Difference in skill distribution (std)

    # Team Names
    team_names: tuple = ("สีแดง (Red)", "สีเขียว (Green)")
