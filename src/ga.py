"""
Genetic Algorithm Engine for Team Division Optimization.
"""
import numpy as np
from typing import List, Tuple, Dict, Any
from config import GAConfig
from fitness import FitnessEvaluator

class GeneticAlgorithm:
    """Executes Genetic Algorithm optimization to find the best team assignment."""

    def __init__(self, evaluator: FitnessEvaluator, config: GAConfig):
        self.evaluator = evaluator
        self.config = config
        self.n = evaluator.n
        self.half_n = self.n // 2
        
        # Set random seed
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            
        self.population: List[np.ndarray] = []
        self.history: Dict[str, List[float]] = {
            "generation": [],
            "best_fitness": [],
            "mean_fitness": [],
            "best_sports_score": [],
            "best_harmonize_score": []
        }
        
    def _create_balanced_chromosome(self) -> np.ndarray:
        """Creates a chromosome with strictly balanced 0s and 1s."""
        chrom = np.zeros(self.n, dtype=int)
        ones_count = self.n - self.half_n
        chrom[:ones_count] = 1
        np.random.shuffle(chrom)
        return chrom

    def _repair_balance(self, chromosome: np.ndarray) -> np.ndarray:
        """Repairs chromosome if 0s and 1s are unbalanced."""
        ones = np.where(chromosome == 1)[0]
        target_ones = self.n - self.half_n
        
        if len(ones) > target_ones:
            # Too many 1s -> flip some random 1s to 0s
            flip_indices = np.random.choice(ones, size=(len(ones) - target_ones), replace=False)
            chromosome[flip_indices] = 0
        elif len(ones) < target_ones:
            # Too many 0s -> flip some random 0s to 1s
            zeros = np.where(chromosome == 0)[0]
            flip_indices = np.random.choice(zeros, size=(target_ones - len(ones)), replace=False)
            chromosome[flip_indices] = 1
            
        return chromosome

    def initialize_population(self):
        """Initializes a population of balanced chromosomes."""
        self.population = [self._create_balanced_chromosome() for _ in range(self.config.population_size)]

    def tournament_selection(self, population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
        """Selects the best chromosome from a random tournament group."""
        indices = np.random.choice(len(population), size=self.config.tournament_size, replace=False)
        best_idx = indices[np.argmax([fitnesses[i] for i in indices])]
        return population[best_idx].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Performs two-point crossover followed by balance repair."""
        if np.random.rand() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
            
        pt1 = np.random.randint(1, self.n - 1)
        pt2 = np.random.randint(pt1, self.n)
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Swap segment
        child1[pt1:pt2] = parent2[pt1:pt2]
        child2[pt1:pt2] = parent1[pt1:pt2]
        
        # Repair children to maintain team size equality
        child1 = self._repair_balance(child1)
        child2 = self._repair_balance(child2)
        
        return child1, child2

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        Applies Swap Mutation to exchange team assignments between two individuals,
        preserving team size balance perfectly.
        """
        if np.random.rand() < self.config.mutation_rate:
            zeros = np.where(chromosome == 0)[0]
            ones = np.where(chromosome == 1)[0]
            
            if len(zeros) > 0 and len(ones) > 0:
                idx_zero = np.random.choice(zeros)
                idx_one = np.random.choice(ones)
                
                # Swap genes
                chromosome[idx_zero] = 1
                chromosome[idx_one] = 0
                
        return chromosome

    def run(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Runs the Genetic Algorithm for the configured number of generations.
        Returns:
            best_chromosome: the optimal binary assignment array
            best_details: evaluation dictionary of the best solution
        """
        self.initialize_population()
        
        best_overall_chrom = None
        best_overall_fitness = -float('inf')
        best_overall_details = None
        
        for gen in range(1, self.config.generations + 1):
            # 1. Evaluate fitness for all chromosomes in population
            eval_results = [self.evaluator.evaluate(chrom) for chrom in self.population]
            fitnesses = [res[0] for res in eval_results]
            details_list = [res[1] for res in eval_results]
            
            # Find best of current generation
            current_best_idx = int(np.argmax(fitnesses))
            current_best_fitness = fitnesses[current_best_idx]
            current_mean_fitness = float(np.mean(fitnesses))
            
            if current_best_fitness > best_overall_fitness:
                best_overall_fitness = current_best_fitness
                best_overall_chrom = self.population[current_best_idx].copy()
                best_overall_details = details_list[current_best_idx]
                
            # Log history
            self.history["generation"].append(gen)
            self.history["best_fitness"].append(current_best_fitness)
            self.history["mean_fitness"].append(current_mean_fitness)
            self.history["best_sports_score"].append(details_list[current_best_idx]["sports_score"])
            self.history["best_harmonize_score"].append(details_list[current_best_idx]["harmonize_score"])
            
            # 2. Elitism: preserve top elite chromosomes
            sorted_indices = np.argsort(fitnesses)[::-1]
            new_population = [self.population[sorted_indices[i]].copy() for i in range(self.config.elitism_count)]
            
            # 3. Create next generation
            while len(new_population) < self.config.population_size:
                p1 = self.tournament_selection(self.population, fitnesses)
                p2 = self.tournament_selection(self.population, fitnesses)
                
                c1, c2 = self.crossover(p1, p2)
                
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                
                new_population.append(c1)
                if len(new_population) < self.config.population_size:
                    new_population.append(c2)
                    
            self.population = new_population
            
        return best_overall_chrom, best_overall_details

