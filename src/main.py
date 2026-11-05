"""
Main Execution Script for Sports Team Division using Genetic Algorithm.
"""
import os
import argparse
import json
import pandas as pd
from config import GAConfig
from data_loader import DataLoader
from fitness import FitnessEvaluator
from ga import GeneticAlgorithm
from visualizer import Visualizer

def run_team_division(
    data_path: str = "data/company_members.csv",
    generations: int = 150,
    population_size: int = 100,
    w_sports: float = 0.50,
    w_harmonize: float = 0.50,
    random_seed: int = 42,
    output_dir: str = "outputs"
):
    print("=" * 70)
    print("  GENETIC ALGORITHM: 2-TEAM SPORTS & HARMONIZATION DIVISION")
    print("=" * 70)
    print(f"Data file       : {data_path}")
    print(f"Generations     : {generations}")
    print(f"Population size : {population_size}")
    print(f"Weights         : Sports={w_sports}, Harmonize={w_harmonize}")
    print(f"Random seed     : {random_seed}")
    print(f"Output directory: {output_dir}")
    print("-" * 70)

    # 1. Load Data
    print(">> Step 1: Loading & standardizing dataset...")
    df = DataLoader.load(data_path)
    print(f"   Loaded {len(df)} members across {df['department'].nunique()} departments.")
    
    # 2. Setup Configuration and Fitness Evaluator
    config = GAConfig(
        population_size=population_size,
        generations=generations,
        weight_sports=w_sports,
        weight_harmonize=w_harmonize,
        random_seed=random_seed
    )
    evaluator = FitnessEvaluator(df, config)
    
    # 3. Execute Genetic Algorithm
    print(">> Step 2: Running Genetic Algorithm optimization...")
    ga = GeneticAlgorithm(evaluator, config)
    best_chrom, best_details = ga.run()
    
    print("-" * 70)
    print(f"   Optimization Complete! Best Fitness: {best_details['total_fitness']} / 100")
    print(f"   - Sports Fairness Score : {best_details['sports_score']} / 100")
    print(f"   - Harmonization Score   : {best_details['harmonize_score']} / 100")
    print(f"   - Mean Skill Team Red   : {best_details['team_0_mean_skill']} (Sum: {best_details['team_0_sum_skill']})")
    print(f"   - Mean Skill Team Green : {best_details['team_1_mean_skill']} (Sum: {best_details['team_1_sum_skill']})")
    print(f"   - Mean Skill Difference : {best_details['skill_mean_diff']} pts")
    print(f"   - Headcount Balance     : {best_details['team_0_count']} vs {best_details['team_1_count']}")
    print("-" * 70)
    
    # 4. Create Result Roster
    assigned_df = df.copy()
    assigned_df["team"] = best_chrom
    assigned_df["team_name"] = assigned_df["team"].map({0: config.team_names[0], 1: config.team_names[1]})
    
    # Sort for readability: by team then department then name
    assigned_df = assigned_df.sort_values(by=["team", "department", "name"]).reset_index(drop=True)
    
    # Save CSV
    os.makedirs(output_dir, exist_ok=True)
    roster_path = os.path.join(output_dir, "team_assignment.csv")
    assigned_df.to_csv(roster_path, index=False, encoding="utf-8-sig")
    print(f">> Step 3: Saved team roster to {roster_path}")

    # Save JSON summary metrics
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(best_details, f, indent=4, ensure_ascii=False)
    print(f"   Saved evaluation metrics to {metrics_path}")

    # 5. Generate Visualizations
    print(">> Step 4: Generating visualizations & HTML report...")
    visualizer = Visualizer(output_dir=output_dir)
    p1 = visualizer.plot_fitness_evolution(ga.history)
    p2 = visualizer.plot_department_distribution(best_details, config.team_names)
    p3 = visualizer.plot_sports_balance(assigned_df, config.team_names)
    p4 = visualizer.generate_html_report(assigned_df, best_details, config.team_names)
    
    print(f"   Generated: {p1}")
    print(f"   Generated: {p2}")
    print(f"   Generated: {p3}")
    print(f"   Generated: {p4}")
    print("=" * 70)
    print("   ALL TASKS COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    
    return assigned_df, best_details

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genetic Algorithm for Sports Team Division")
    parser.add_argument("--data", type=str, default="data/company_members.csv", help="Path to input CSV file")
    parser.add_argument("--generations", type=int, default=150, help="Number of GA generations")
    parser.add_argument("--pop_size", type=int, default=100, help="Population size")
    parser.add_argument("--w_sports", type=float, default=0.50, help="Weight for sports fairness")
    parser.add_argument("--w_harmonize", type=float, default=0.50, help="Weight for harmonization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")

    args = parser.parse_args()
    run_team_division(
        data_path=args.data,
        generations=args.generations,
        population_size=args.pop_size,
        w_sports=args.w_sports,
        w_harmonize=args.w_harmonize,
        random_seed=args.seed,
        output_dir=args.output_dir
    )

