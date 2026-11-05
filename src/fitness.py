"""
Fitness Evaluation Engine for Sports Engagement and Department Harmonization.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from config import GAConfig

class FitnessEvaluator:
    """Evaluates chromosomes based on Sports Engagement and Cross-Department Harmonization."""
    
    def __init__(self, df: pd.DataFrame, config: GAConfig):
        self.df = df.copy()
        self.config = config
        self.n = len(df)
        self.skills = self.df['sports_skill'].values.astype(float)
        self.departments = self.df['department'].values
        self.unique_depts, self.dept_indices = np.unique(self.departments, return_inverse=True)
        self.num_depts = len(self.unique_depts)
        
        # Skill scale
        self.skill_min = float(np.min(self.skills))
        self.skill_max = float(np.max(self.skills))
        self.skill_range = max(1.0, self.skill_max - self.skill_min)
        
        # Theoretical max entropy for departments
        self.max_entropy = np.log2(self.num_depts) if self.num_depts > 1 else 1.0

    def evaluate(self, chromosome: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluates a single chromosome (binary array of size N).
        Returns:
            fitness: scalar fitness score
            details: comprehensive dictionary of evaluation metrics
        """
        mask_0 = (chromosome == 0)
        mask_1 = (chromosome == 1)
        
        count_0 = int(np.sum(mask_0))
        count_1 = int(np.sum(mask_1))
        
        # Handle edge case where a team is empty
        if count_0 == 0 or count_1 == 0:
            return -1000.0, {"error": "Empty team"}
            
        skills_0 = self.skills[mask_0]
        skills_1 = self.skills[mask_1]
        
        # 1. Sports Engagement & Fairness Calculation
        mean_0 = float(np.mean(skills_0))
        mean_1 = float(np.mean(skills_1))
        sum_0 = float(np.sum(skills_0))
        sum_1 = float(np.sum(skills_1))
        std_0 = float(np.std(skills_0)) if count_0 > 1 else 0.0
        std_1 = float(np.std(skills_1)) if count_1 > 1 else 0.0
        
        mean_diff = abs(mean_0 - mean_1)
        std_diff = abs(std_0 - std_1)
        
        # Normalized Sports Fairness Score (0 to 100)
        norm_mean_err = min(1.0, mean_diff / self.skill_range)
        norm_std_err = min(1.0, std_diff / self.skill_range)
        
        sports_score = 100.0 * (1.0 - (
            self.config.sports_mean_weight * norm_mean_err +
            self.config.sports_std_weight * norm_std_err
        ))
        sports_score = max(0.0, sports_score)
        
        # 2. Cross-Department Harmonization Calculation
        # A) Department Parity (Balanced split of each department)
        dept_counts_0 = np.bincount(self.dept_indices[mask_0], minlength=self.num_depts)
        dept_counts_1 = np.bincount(self.dept_indices[mask_1], minlength=self.num_depts)
        dept_totals = dept_counts_0 + dept_counts_1
        
        # Ideal discrepancy per department is (total % 2)
        excess_discrepancy = 0
        for d in range(self.num_depts):
            diff = abs(dept_counts_0[d] - dept_counts_1[d])
            min_allowed_diff = dept_totals[d] % 2
            excess_discrepancy += max(0, diff - min_allowed_diff)
            
        parity_score = 100.0 * (1.0 - (excess_discrepancy / self.n))
        parity_score = max(0.0, parity_score)
        
        # B) Team Diversity (Shannon Entropy)
        p0 = dept_counts_0 / count_0
        p1 = dept_counts_1 / count_1
        
        entropy_0 = -np.sum([p * np.log2(p) for p in p0 if p > 0])
        entropy_1 = -np.sum([p * np.log2(p) for p in p1 if p > 0])
        
        norm_entropy_0 = entropy_0 / self.max_entropy
        norm_entropy_1 = entropy_1 / self.max_entropy
        entropy_score = 100.0 * (0.5 * norm_entropy_0 + 0.5 * norm_entropy_1)
        entropy_score = max(0.0, min(100.0, entropy_score))
        
        harmonize_score = (
            self.config.harmonize_parity_weight * parity_score +
            self.config.harmonize_entropy_weight * entropy_score
        )
        
        # 3. Team Size Imbalance Penalty
        size_diff = abs(count_0 - count_1)
        size_penalty = self.config.penalty_size_weight * size_diff
        
        # Combined Total Fitness
        total_fitness = (
            self.config.weight_sports * sports_score +
            self.config.weight_harmonize * harmonize_score -
            size_penalty
        )
        
        details = {
            "total_fitness": round(total_fitness, 2),
            "sports_score": round(sports_score, 2),
            "harmonize_score": round(harmonize_score, 2),
            "parity_score": round(parity_score, 2),
            "entropy_score": round(entropy_score, 2),
            "team_0_count": count_0,
            "team_1_count": count_1,
            "team_0_mean_skill": round(mean_0, 2),
            "team_1_mean_skill": round(mean_1, 2),
            "team_0_sum_skill": round(sum_0, 2),
            "team_1_sum_skill": round(sum_1, 2),
            "team_0_std_skill": round(std_0, 2),
            "team_1_std_skill": round(std_1, 2),
            "skill_mean_diff": round(mean_diff, 2),
            "skill_sum_diff": round(abs(sum_0 - sum_1), 2),
            "size_difference": size_diff,
            "dept_counts_team_0": dict(zip(self.unique_depts, [int(x) for x in dept_counts_0])),
            "dept_counts_team_1": dict(zip(self.unique_depts, [int(x) for x in dept_counts_1])),
        }
        
        return total_fitness, details

