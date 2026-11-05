"""
Unit Tests for Genetic Algorithm Team Division System.
"""
import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import GAConfig
from data_loader import DataLoader
from fitness import FitnessEvaluator
from ga import GeneticAlgorithm

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'id': [f'P{i:02d}' for i in range(10)],
        'name': [f'Person_{i}' for i in range(10)],
        'department': ['IT', 'IT', 'IT', 'IT', 'HR', 'HR', 'Sales', 'Sales', 'Sales', 'Sales'],
        'sports_skill': [50, 60, 70, 80, 40, 90, 55, 65, 75, 85]
    })

def test_data_loader_company():
    df = DataLoader.load('data/company_members.csv')
    assert len(df) == 40
    assert set(df.columns) == {'id', 'name', 'department', 'sports_skill'}
    assert df['sports_skill'].min() >= 0

def test_chromosome_balance(sample_df):
    config = GAConfig()
    evaluator = FitnessEvaluator(sample_df, config)
    ga = GeneticAlgorithm(evaluator, config)
    
    chrom = ga._create_balanced_chromosome()
    assert len(chrom) == 10
    assert np.sum(chrom == 0) == 5
    assert np.sum(chrom == 1) == 5

def test_repair_unbalanced_chromosome(sample_df):
    config = GAConfig()
    evaluator = FitnessEvaluator(sample_df, config)
    ga = GeneticAlgorithm(evaluator, config)
    
    # Intentionally unbalanced: 8 ones and 2 zeros
    unbalanced = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    repaired = ga._repair_balance(unbalanced)
    
    assert np.sum(repaired == 0) == 5
    assert np.sum(repaired == 1) == 5

def test_swap_mutation_preserves_size(sample_df):
    config = GAConfig(mutation_rate=1.0)
    evaluator = FitnessEvaluator(sample_df, config)
    ga = GeneticAlgorithm(evaluator, config)
    
    chrom = ga._create_balanced_chromosome()
    mutated = ga.mutate(chrom.copy())
    
    assert np.sum(mutated == 0) == 5
    assert np.sum(mutated == 1) == 5

def test_fitness_bounds(sample_df):
    config = GAConfig()
    evaluator = FitnessEvaluator(sample_df, config)
    
    chrom = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    fitness, details = evaluator.evaluate(chrom)
    
    assert 0 <= details['sports_score'] <= 100
    assert 0 <= details['harmonize_score'] <= 100
    assert details['team_0_count'] == 5
    assert details['team_1_count'] == 5

def test_ga_execution(sample_df):
    config = GAConfig(population_size=30, generations=20, random_seed=42)
    evaluator = FitnessEvaluator(sample_df, config)
    ga = GeneticAlgorithm(evaluator, config)
    
    best_chrom, best_details = ga.run()
    
    assert len(best_chrom) == 10
    assert best_details['team_0_count'] == 5
    assert best_details['team_1_count'] == 5
    assert best_details['total_fitness'] > 70.0
