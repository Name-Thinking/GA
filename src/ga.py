"""
==============================================================================
 อัลกอริทึมทางพันธุกรรม (Genetic Algorithm - GA)
==============================================================================
หน้าที่: ค้นหาการจับคู่สีที่ดีที่สุด (Optimal Team Assignment)
ขั้นตอนหลักของ GA:
  1. สร้างประชากรรุ่นแรก (Initialization)
  2. ประเมินความเหมาะสม (Evaluation)
  3. คัดเลือกพ่อแม่พันธุ์ (Selection)
  4. ผสมพันธุ์แลกเปลี่ยนยีน (Crossover)
  5. กลายพันธุ์ (Mutation)
  6. สืบทอดโครโมโซมที่ดีที่สุดไปรุ่นถัดไป (Elitism)
==============================================================================
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from config import GAConfig
from fitness import FitnessEvaluator


class GeneticAlgorithm:
    """คลาสหลักในการประมวลผล Genetic Algorithm"""

    def __init__(self, evaluator: FitnessEvaluator, config: GAConfig):
        self.evaluator = evaluator
        self.config = config
        self.n = evaluator.total_members     # จำนวนคนทั้งหมด
        self.half_n = self.n // 2            # โควตาครึ่งหนึ่งสำหรับแต่ละสี
        
        # กำหนด Random Seed เพื่อให้ผลลัพธ์สามารถทดสอบซ้ำได้ (Reproducibility)
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
            
        self.population: List[np.ndarray] = []
        
        # ประวัติการทำงานในแต่ละรุ่น (ใช้สำหรับพล็อตดูกราฟ)
        self.history: Dict[str, List[float]] = {
            "generation": [],
            "best_fitness": [],
            "mean_fitness": [],
            "best_sports_score": [],
            "best_harmonize_score": []
        }
        
    def _create_balanced_chromosome(self) -> np.ndarray:
        """
        [ขั้นตอนที่ 1] สร้างโครโมโซมเริ่มต้นที่มีขนาดสมดุล 50:50 ทันที
        ตัวอย่าง: มี 40 คน -> สร้างเลข 0 จำนวน 20 ตัว และเลข 1 จำนวน 20 ตัว แล้วสับเปลี่ยนตำแหน่ง
        """
        chrom = np.zeros(self.n, dtype=int)
        ones_count = self.n - self.half_n
        chrom[:ones_count] = 1
        np.random.shuffle(chrom)
        return chrom

    def _repair_balance(self, chromosome: np.ndarray) -> np.ndarray:
        """
        [ฟังก์ชันช่วยซ่อมแซม] ปรับแก้โครโมโซมให้มีจำนวน 0 และ 1 เท่ากันเสมอ
        ป้องกันปัญหาทีมใดทีมหนึ่งมีจำนวนคนมากกว่าอีกทีม
        """
        ones = np.where(chromosome == 1)[0]
        target_ones = self.n - self.half_n
        
        if len(ones) > target_ones:
            # ถ้ามีสีเขียว (1) มากเกินไป -> สุ่มเปลี่ยน 1 เป็น 0
            excess = len(ones) - target_ones
            flip_indices = np.random.choice(ones, size=excess, replace=False)
            chromosome[flip_indices] = 0
            
        elif len(ones) < target_ones:
            # ถ้ามีสีแดง (0) มากเกินไป -> สุ่มเปลี่ยน 0 เป็น 1
            zeros = np.where(chromosome == 0)[0]
            deficit = target_ones - len(ones)
            flip_indices = np.random.choice(zeros, size=deficit, replace=False)
            chromosome[flip_indices] = 1
            
        return chromosome

    def initialize_population(self):
        """สร้างประชากรรุ่นแรกตามขนาด population_size ที่ตั้งไว้"""
        self.population = [self._create_balanced_chromosome() for _ in range(self.config.population_size)]

    def tournament_selection(self, population: List[np.ndarray], fitnesses: List[float]) -> np.ndarray:
        """
        [ขั้นตอนที่ 2: Selection] คัดเลือกพ่อแม่พันธุ์ด้วยวิธี Tournament Selection
        สุ่มผู้เข้าแข่งขันมา k ตัว แล้วเลือกตัวที่มี Fitness สูงที่สุดเพื่อส่งต่อไปผสมพันธุ์
        """
        candidate_indices = np.random.choice(len(population), size=self.config.tournament_size, replace=False)
        best_idx = candidate_indices[np.argmax([fitnesses[i] for i in candidate_indices])]
        return population[best_idx].copy()

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        [ขั้นตอนที่ 3: Crossover] การผสมข้ามสายพันธุ์ด้วยวิธี Two-Point Crossover
        สุ่มจุดตัด 2 จุด แล้วสลับชิ้นส่วนยีนระหว่างพ่อกับแม่เพื่อให้ได้ลูกผสม 2 ตัวใหม่
        """
        if np.random.rand() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()
            
        # สุ่มจุดตัด 2 จุด
        pt1 = np.random.randint(1, self.n - 1)
        pt2 = np.random.randint(pt1, self.n)
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # สลับท่อนยีนระหว่างจุด pt1 และ pt2
        child1[pt1:pt2] = parent2[pt1:pt2]
        child2[pt1:pt2] = parent1[pt1:pt2]
        
        # ซ่อมแซมลูกทั้งสองตัวให้จำนวนสมาชิกทั้ง 2 สียังคงเท่ากัน
        child1 = self._repair_balance(child1)
        child2 = self._repair_balance(child2)
        
        return child1, child2

    def mutate(self, chromosome: np.ndarray) -> np.ndarray:
        """
        [ขั้นตอนที่ 4: Mutation] การกลายพันธุ์ด้วยวิธี Swap Mutation
        สุ่มเลือกคนสีแดง (0) หนึ่งคน และคนสีเขียว (1) หนึ่งคน แล้วสลับสีกัน
        วิธีนี้ทำให้ขนาดของแต่ละทีมยังคงเท่าเดิม 100% เสมอ
        """
        if np.random.rand() < self.config.mutation_rate:
            zeros = np.where(chromosome == 0)[0]
            ones = np.where(chromosome == 1)[0]
            
            if len(zeros) > 0 and len(ones) > 0:
                idx_zero = np.random.choice(zeros)
                idx_one = np.random.choice(ones)
                
                # สลับสีของผู้เล่น 2 คนนี้
                chromosome[idx_zero] = 1
                chromosome[idx_one] = 0
                
        return chromosome

    def run(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        [ขั้นตอนหลัก] ลูปวิวัฒนาการตามจำนวนรุ่น (Generations)
        """
        self.initialize_population()
        
        best_overall_chrom = None
        best_overall_fitness = -float('inf')
        best_overall_details = None
        
        for gen in range(1, self.config.generations + 1):
            # 1. ประเมินคะแนน Fitness ของประชากรทุกคนในรุ่นปัจจุบัน
            eval_results = [self.evaluator.evaluate(chrom) for chrom in self.population]
            fitnesses = [res[0] for res in eval_results]
            details_list = [res[1] for res in eval_results]
            
            # หาโครโมโซมที่เก่งที่สุดในรุ่นนี้
            current_best_idx = int(np.argmax(fitnesses))
            current_best_fitness = fitnesses[current_best_idx]
            current_mean_fitness = float(np.mean(fitnesses))
            
            # บันทึกโครโมโซมที่ดีที่สุดตลอดกาล (Best of All Time)
            if current_best_fitness > best_overall_fitness:
                best_overall_fitness = current_best_fitness
                best_overall_chrom = self.population[current_best_idx].copy()
                best_overall_details = details_list[current_best_idx]
                
            # บันทึกสถิติเพื่อนำไปพล็อตกราฟ
            self.history["generation"].append(gen)
            self.history["best_fitness"].append(current_best_fitness)
            self.history["mean_fitness"].append(current_mean_fitness)
            self.history["best_sports_score"].append(details_list[current_best_idx]["sports_score"])
            self.history["best_harmonize_score"].append(details_list[current_best_idx]["harmonize_score"])
            
            # 2. Elitism: คัดเลือกตัวท็อปที่ดีที่สุดเก็บไว้ ไม่ให้สูญหายในรุ่นถัดไป
            sorted_indices = np.argsort(fitnesses)[::-1]
            new_population = [self.population[sorted_indices[i]].copy() for i in range(self.config.elitism_count)]
            
            # 3. ผลิตสมาชิกรุ่นใหม่จนเต็มจำนวนประชากร
            while len(new_population) < self.config.population_size:
                # คัดเลือกพ่อแม่
                parent1 = self.tournament_selection(self.population, fitnesses)
                parent2 = self.tournament_selection(self.population, fitnesses)
                
                # ผสมพันธุ์
                child1, child2 = self.crossover(parent1, parent2)
                
                # กลายพันธุ์
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                # เพิ่มลูกเข้าสู่ประชากรรุ่นใหม่
                new_population.append(child1)
                if len(new_population) < self.config.population_size:
                    new_population.append(child2)
                    
            # เปลี่ยนผ่านสู่รุ่นถัดไป
            self.population = new_population
            
        return best_overall_chrom, best_overall_details
