"""
==============================================================================
 โมดูลคำนวณค่าความเหมาะสม (Fitness Function)
==============================================================================
หน้าที่: ประเมินว่า "การจัดทีม" (Chromosome) มีความเหมาะสมมากน้อยเพียงใด
โดยวัดจาก 2 เกณฑ์หลักตามโจทย์:
  1. ความแฟนเนจทางด้านกีฬา (Sports Fairness & Engagement)
  2. การสร้างความรู้จักข้ามแผนก (Cross-Department Harmonization)
==============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from config import GAConfig


class FitnessEvaluator:
    """คลาสสำหรับประเมินคะแนนความเหมาะสม (Fitness) ของโครโมโซม"""

    def __init__(self, df: pd.DataFrame, config: GAConfig):
        self.df = df.copy()
        self.config = config
        self.total_members = len(df)
        
        # ดึงคะแนนสกิลกีฬาและแผนกของทุกคนออกมาเป็น Array
        self.skills = self.df['sports_skill'].values.astype(float)
        self.departments = self.df['department'].values
        
        # หาแผนกทั้งหมดที่มีในข้อมูล
        self.unique_depts, self.dept_indices = np.unique(self.departments, return_inverse=True)
        self.num_depts = len(self.unique_depts)
        
        # ช่วงคะแนนกีฬา (Max - Min) ใช้สำหรับปรับสเกลคะแนนให้อยู่ในรูปเปอร์เซ็นต์
        self.skill_min = float(np.min(self.skills))
        self.skill_max = float(np.max(self.skills))
        self.skill_range = max(1.0, self.skill_max - self.skill_min)
        
        # ค่าความหลากหลายสูงสุดที่เป็นไปได้ (Theoretical Max Entropy)
        self.max_entropy = np.log2(self.num_depts) if self.num_depts > 1 else 1.0

    def evaluate(self, chromosome: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        ประเมินโครโมโซม 1 ชุด:
          โครโมโซม [0, 1, 0, 1, ...]
          0 = ทีมสีแดง (Red)
          1 = ทีมสีเขียว (Green)
        """
        # แยกสมาชิกตามสี: 0 = สีแดง, 1 = สีเขียว
        is_red = (chromosome == 0)
        is_green = (chromosome == 1)
        
        count_red = int(np.sum(is_red))
        count_green = int(np.sum(is_green))
        
        # กรณีฉุกเฉิน: ถ้ามีทีมใดทีมหนึ่งว่างเปล่า ให้คะแนนติดลบหนักๆ
        if count_red == 0 or count_green == 0:
            return -1000.0, {"error": "ทีมว่างเปล่า"}
            
        # ดึงคะแนนกีฬาของสมาชิกแต่ละสี
        skills_red = self.skills[is_red]
        skills_green = self.skills[is_green]
        
        # ====================================================================
        # เกณฑ์ที่ 1: ความแฟนเนจทางด้านกีฬา (Sports Fairness Score: 0 - 100)
        # ====================================================================
        # วัตถุประสงค์: คะแนนเฉลี่ยของทั้ง 2 สีต้องใกล้เคียงกันที่สุด เพื่อให้แข่งสนุก แฟร์
        mean_red = float(np.mean(skills_red))
        mean_green = float(np.mean(skills_green))
        sum_red = float(np.sum(skills_red))
        sum_green = float(np.sum(skills_green))
        
        # ความแปรปรวน (เพื่อดูว่ากระจายคนเก่งกับมือใหม่สมดุลหรือไม่)
        std_red = float(np.std(skills_red)) if count_red > 1 else 0.0
        std_green = float(np.std(skills_green)) if count_green > 1 else 0.0
        
        # หาผลต่างระหว่าง 2 ทีม
        mean_diff = abs(mean_red - mean_green)
        std_diff = abs(std_red - std_green)
        
        # แปลงผลต่างให้อยู่ในสเกล 0 ถึง 1 (ยิ่งผลต่างน้อย = ยิ่งดี)
        norm_mean_error = min(1.0, mean_diff / self.skill_range)
        norm_std_error = min(1.0, std_diff / self.skill_range)
        
        # คำนวณคะแนนกีฬาเต็ม 100 (ถ้าผลต่างเป็น 0 จะได้ 100 เต็ม)
        sports_score = 100.0 * (1.0 - (
            self.config.sports_mean_weight * norm_mean_error +
            self.config.sports_std_weight * norm_std_error
        ))
        sports_score = max(0.0, sports_score)
        
        # ====================================================================
        # เกณฑ์ที่ 2: ความรู้จักข้ามแผนก (Cross-Department Harmonization: 0 - 100)
        # ====================================================================
        # ส่วนย่อย A: การกระจายคนแต่ละแผนกให้ลง 2 สีเท่าๆ กัน (Department Parity)
        dept_counts_red = np.bincount(self.dept_indices[is_red], minlength=self.num_depts)
        dept_counts_green = np.bincount(self.dept_indices[is_green], minlength=self.num_depts)
        dept_totals = dept_counts_red + dept_counts_green
        
        excess_discrepancy = 0
        for d in range(self.num_depts):
            diff = abs(dept_counts_red[d] - dept_counts_green[d])
            min_allowed = dept_totals[d] % 2  # ถ้าคนเป็นเลขคี่ อนุญาตให้ต่างกันได้ 1 คน
            excess_discrepancy += max(0, diff - min_allowed)
            
        parity_score = 100.0 * (1.0 - (excess_discrepancy / self.total_members))
        parity_score = max(0.0, parity_score)
        
        # ส่วนย่อย B: ความหลากหลายของแผนกภายในทีม (Team Diversity / Shannon Entropy)
        p_red = dept_counts_red / count_red
        p_green = dept_counts_green / count_green
        
        entropy_red = -np.sum([p * np.log2(p) for p in p_red if p > 0])
        entropy_green = -np.sum([p * np.log2(p) for p in p_green if p > 0])
        
        norm_entropy_red = entropy_red / self.max_entropy
        norm_entropy_green = entropy_green / self.max_entropy
        entropy_score = 100.0 * (0.5 * norm_entropy_red + 0.5 * norm_entropy_green)
        entropy_score = max(0.0, min(100.0, entropy_score))
        
        # รวมคะแนน Harmonize
        harmonize_score = (
            self.config.harmonize_parity_weight * parity_score +
            self.config.harmonize_entropy_weight * entropy_score
        )
        
        # ====================================================================
        # บทลงโทษหากขนาด 2 ทีมไม่เท่ากัน (Size Imbalance Penalty)
        # ====================================================================
        size_diff = abs(count_red - count_green)
        size_penalty = self.config.penalty_size_weight * size_diff
        
        # ====================================================================
        # สรุปคะแนน Fitness รวม (เต็ม 100)
        # ====================================================================
        total_fitness = (
            self.config.weight_sports * sports_score +
            self.config.weight_harmonize * harmonize_score -
            size_penalty
        )
        
        # จัดเก็บข้อมูลสำหรับแสดงผล
        details = {
            "total_fitness": round(total_fitness, 2),
            "sports_score": round(sports_score, 2),
            "harmonize_score": round(harmonize_score, 2),
            "parity_score": round(parity_score, 2),
            "entropy_score": round(entropy_score, 2),
            "team_0_count": count_red,
            "team_1_count": count_green,
            "team_0_mean_skill": round(mean_red, 2),
            "team_1_mean_skill": round(mean_green, 2),
            "team_0_sum_skill": round(sum_red, 2),
            "team_1_sum_skill": round(sum_green, 2),
            "team_0_std_skill": round(std_red, 2),
            "team_1_std_skill": round(std_green, 2),
            "skill_mean_diff": round(mean_diff, 2),
            "skill_sum_diff": round(abs(sum_red - sum_green), 2),
            "size_difference": size_diff,
            "dept_counts_team_0": dict(zip(self.unique_depts, [int(x) for x in dept_counts_red])),
            "dept_counts_team_1": dict(zip(self.unique_depts, [int(x) for x in dept_counts_green])),
        }
        
        return total_fitness, details
