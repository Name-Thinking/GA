"""
Visualization Module for GA Team Division Results.
Generates charts for Fitness Evolution, Department Distribution, and Sports Skill Balance.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, Any, List

# Set clean aesthetic styling with Thai font fallback support
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Leelawadee UI', 'Tahoma', 'Angsana New', 'Microsoft Sans Serif', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class Visualizer:
    """Generates visualizations for GA optimization and evaluation."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_fitness_evolution(self, history: Dict[str, List[float]], save_name: str = "fitness_evolution.png") -> str:
        """Plots the GA fitness evolution over generations."""
        plt.figure(figsize=(10, 6))
        gens = history["generation"]
        
        plt.plot(gens, history["best_fitness"], label="Best Total Fitness", color="#D32F2F", linewidth=2.5)
        plt.plot(gens, history["mean_fitness"], label="Mean Population Fitness", color="#1976D2", linestyle="--", linewidth=1.8)
        plt.plot(gens, history["best_sports_score"], label="Sports Fairness Score", color="#388E3C", linestyle=":", linewidth=2)
        plt.plot(gens, history["best_harmonize_score"], label="Harmonize Score", color="#7B1FA2", linestyle="-.", linewidth=2)
        
        plt.title("Genetic Algorithm Convergence & Optimization Progress", fontsize=15, pad=15, fontweight='bold')
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Score / Fitness (0 - 100)", fontsize=12)
        plt.ylim(min(history["mean_fitness"]) - 5, 105)
        plt.legend(loc="lower right", frameon=True, fontsize=11)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path

    def plot_department_distribution(self, details: Dict[str, Any], team_names: tuple, save_name: str = "department_distribution.png") -> str:
        """Plots grouped bar chart of department counts in both teams."""
        depts = list(details["dept_counts_team_0"].keys())
        counts_0 = [details["dept_counts_team_0"][d] for d in depts]
        counts_1 = [details["dept_counts_team_1"][d] for d in depts]
        
        x = np.arange(len(depts))
        width = 0.35
        
        plt.figure(figsize=(11, 6))
        rects1 = plt.bar(x - width/2, counts_0, width, label=team_names[0], color="#E53935", edgecolor="black", alpha=0.85)
        rects2 = plt.bar(x + width/2, counts_1, width, label=team_names[1], color="#43A047", edgecolor="black", alpha=0.85)
        
        plt.ylabel("Number of Members", fontsize=12)
        plt.title("Cross-Department Harmonization: Distribution across Teams", fontsize=15, pad=15, fontweight='bold')
        plt.xticks(x, depts, rotation=25, ha="right", fontsize=11)
        plt.legend(fontsize=11)
        
        # Add values on top of bars
        for rect in rects1:
            h = rect.get_height()
            plt.annotate(f"{h}", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
        for rect in rects2:
            h = rect.get_height()
            plt.annotate(f"{h}", xy=(rect.get_x() + rect.get_width() / 2, h),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
                         
        plt.ylim(0, max(max(counts_0), max(counts_1)) + 2)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path

    def plot_sports_balance(self, assigned_df: pd.DataFrame, team_names: tuple, save_name: str = "sports_balance.png") -> str:
        """Plots distribution of sports skills and summary metrics for both teams."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colors = ["#E53935", "#43A047"]
        
        # Subplot 1: Boxplot & Strip plot
        sns.boxplot(ax=axes[0], data=assigned_df, x="team_name", y="sports_skill", hue="team_name", palette=colors, width=0.45, legend=False)
        sns.stripplot(ax=axes[0], data=assigned_df, x="team_name", y="sports_skill", color="black", alpha=0.6, jitter=0.2, size=7)
        axes[0].set_title("Sports Skill Distribution (Fairness Check)", fontsize=13, fontweight='bold')
        axes[0].set_xlabel("Team", fontsize=11)
        axes[0].set_ylabel("Sports Skill / Capability", fontsize=11)
        
        # Subplot 2: Bar chart comparison of Mean and Standard Deviation
        stats = assigned_df.groupby("team_name")["sports_skill"].agg(["mean", "std", "count"]).reset_index()
        x = np.arange(len(stats))
        width = 0.35
        
        rects1 = axes[1].bar(x - width/2, stats["mean"], width, label="Mean Skill", color="#1E88E5", edgecolor="black", alpha=0.85)
        rects2 = axes[1].bar(x + width/2, stats["std"], width, label="Std Dev (Spread)", color="#FB8C00", edgecolor="black", alpha=0.85)
        
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(stats["team_name"], fontsize=11)
        axes[1].set_title("Comparison of Skill Means & Variance", fontsize=13, fontweight='bold')
        axes[1].set_ylabel("Score", fontsize=11)
        axes[1].legend(fontsize=10)
        
        for rect in rects1:
            h = rect.get_height()
            axes[1].annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
        for rect in rects2:
            h = rect.get_height()
            axes[1].annotate(f"{h:.1f}", xy=(rect.get_x() + rect.get_width() / 2, h),
                             xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')
                             
        plt.tight_layout()
        save_path = os.path.join(self.output_dir, save_name)
        plt.savefig(save_path, dpi=300)
        plt.close()
        return save_path

    def generate_html_report(self, assigned_df: pd.DataFrame, details: Dict[str, Any], team_names: tuple, save_name: str = "report.html") -> str:
        """Generates a self-contained HTML dashboard of the solution."""
        team_0_members = assigned_df[assigned_df['team'] == 0].to_dict('records')
        team_1_members = assigned_df[assigned_df['team'] == 1].to_dict('records')
        
        html_content = f"""<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>รายงานผลการแบ่งทีมกีฬาด้วย Genetic Algorithm (GA)</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f4f6f9; color: #333; margin: 0; padding: 25px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #1e3c72, #2a5298); color: white; padding: 25px; border-radius: 12px; margin-bottom: 25px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
        .header h1 {{ margin: 0 0 10px 0; font-size: 26px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 15px; }}
        
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 20px; margin-bottom: 25px; }}
        .card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center; border-top: 4px solid #2a5298; }}
        .card.sports {{ border-top-color: #388E3C; }}
        .card.harmonize {{ border-top-color: #7B1FA2; }}
        .card.fitness {{ border-top-color: #D32F2F; }}
        .card .title {{ font-size: 14px; color: #666; margin-bottom: 8px; text-transform: uppercase; font-weight: 600; }}
        .card .value {{ font-size: 32px; font-weight: bold; color: #111; }}
        .card .subtext {{ font-size: 13px; color: #777; margin-top: 5px; }}
        
        .teams-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 25px; margin-bottom: 25px; }}
        @media (max-width: 850px) {{ .teams-grid {{ grid-template-columns: 1fr; }} }}
        
        .team-box {{ background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
        .team-box.red .team-header {{ background-color: #E53935; color: white; }}
        .team-box.green .team-header {{ background-color: #43A047; color: white; }}
        .team-header {{ padding: 15px 20px; display: flex; justify-content: space-between; align-items: center; }}
        .team-header h2 {{ margin: 0; font-size: 18px; }}
        .team-header span {{ font-size: 14px; background: rgba(255,255,255,0.25); padding: 4px 10px; border-radius: 20px; }}
        
        .team-body {{ padding: 15px; max-height: 500px; overflow-y: auto; }}
        table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background-color: #f8f9fa; color: #555; position: sticky; top: 0; }}
        .badge {{ display: inline-block; padding: 3px 8px; border-radius: 12px; font-size: 11px; background: #e9ecef; color: #495057; font-weight: 600; }}
        
        .section-title {{ font-size: 20px; margin: 30px 0 15px 0; color: #2a5298; border-bottom: 2px solid #e2e8f0; padding-bottom: 8px; }}
        .img-grid {{ display: grid; grid-template-columns: 1fr; gap: 20px; margin-bottom: 30px; }}
        .img-card {{ background: white; padding: 15px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center; }}
        .img-card img {{ max-width: 100%; height: auto; border-radius: 6px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>รายงานผลการจัดทีมกีฬา 2 สี (Genetic Algorithm)</h1>
            <p>สมดุลความแฟนเนจทางกีฬา (Sports Engagement & Fairness) และสร้างความรู้จักข้ามแผนก (Harmonization)</p>
        </div>

        <div class="metric-grid">
            <div class="card fitness">
                <div class="title">Total GA Fitness</div>
                <div class="value">{details['total_fitness']} <span style="font-size: 18px;">/ 100</span></div>
                <div class="subtext">คะแนนฟิตเนสรวมของอัลกอริทึม</div>
            </div>
            <div class="card sports">
                <div class="title">Sports Fairness Score</div>
                <div class="value">{details['sports_score']} <span style="font-size: 18px;">/ 100</span></div>
                <div class="subtext">ผลต่างค่าเฉลี่ยสกิล: {details['skill_mean_diff']} แต้ม</div>
            </div>
            <div class="card harmonize">
                <div class="title">Harmonize Score</div>
                <div class="value">{details['harmonize_score']} <span style="font-size: 18px;">/ 100</span></div>
                <div class="subtext">ความกระจายแผนก (Entropy: {details['entropy_score']}%)</div>
            </div>
            <div class="card">
                <div class="title">Team Size Balance</div>
                <div class="value">{details['team_0_count']} : {details['team_1_count']}</div>
                <div class="subtext">ความต่างจำนวนคน: {details['size_difference']} คน</div>
            </div>
        </div>

        <div class="teams-grid">
            <div class="team-box red">
                <div class="team-header">
                    <h2>{team_names[0]}</h2>
                    <span>สมาชิก: {details['team_0_count']} คน | ค่าเฉลี่ยสกิล: {details['team_0_mean_skill']}</span>
                </div>
                <div class="team-body">
                    <table>
                        <thead>
                            <tr><th>รหัส</th><th>ชื่อ-นามสกุล</th><th>แผนก</th><th>สกิลกีฬา</th></tr>
                        </thead>
                        <tbody>
                            {"".join([f"<tr><td>{m['id']}</td><td>{m['name']}</td><td><span class='badge'>{m['department']}</span></td><td><b>{m['sports_skill']}</b></td></tr>" for m in team_0_members])}
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="team-box green">
                <div class="team-header">
                    <h2>{team_names[1]}</h2>
                    <span>สมาชิก: {details['team_1_count']} คน | ค่าเฉลี่ยสกิล: {details['team_1_mean_skill']}</span>
                </div>
                <div class="team-body">
                    <table>
                        <thead>
                            <tr><th>รหัส</th><th>ชื่อ-นามสกุล</th><th>แผนก</th><th>สกิลกีฬา</th></tr>
                        </thead>
                        <tbody>
                            {"".join([f"<tr><td>{m['id']}</td><td>{m['name']}</td><td><span class='badge'>{m['department']}</span></td><td><b>{m['sports_skill']}</b></td></tr>" for m in team_1_members])}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <h2 class="section-title">การแสดงผลเชิงภาพ (Visualizations)</h2>
        <div class="img-grid">
            <div class="img-card">
                <h3>1. กราฟการลู่เข้าของค่าฟิตเนส (Fitness Convergence)</h3>
                <img src="fitness_evolution.png" alt="Fitness Evolution">
            </div>
            <div class="img-card">
                <h3>2. การกระจายตัวของสมาชิกแต่ละแผนกใน 2 สี (Harmonization)</h3>
                <img src="department_distribution.png" alt="Department Distribution">
            </div>
            <div class="img-card">
                <h3>3. เปรียบเทียบสมรรถภาพและการกระจายตัวของทักษะกีฬา (Sports Fairness)</h3>
                <img src="sports_balance.png" alt="Sports Skill Balance">
            </div>
        </div>
    </div>
</body>
</html>
"""
        save_path = os.path.join(self.output_dir, save_name)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return save_path
