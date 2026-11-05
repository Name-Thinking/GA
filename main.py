"""
=============================================================
 ระบบแบ่งทีมกีฬา 2 สี (สีแดง / สีเขียว) ด้วย Genetic Algorithm (GA)
=============================================================
 เป้าหมายหลักของโจทย์:
   1. กีฬาสูสี (Sports Fairness): ค่าเฉลี่ยสกิลกีฬาของ 2 สีต้องใกล้เคียงกันที่สุด
   2. รู้จักข้ามแผนก (Harmonize) : แต่ละแผนกต้องแบ่งลง 2 สีเท่าๆ กัน
=============================================================
"""

import random
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# 1. โหลดข้อมูลสมาชิก
# -----------------------------------------------------------
df = pd.read_csv("data/company_members.csv")

# แปลงเป็น List ธรรมดาเพื่อให้วนลูปและเข้าใจง่าย
skills = df['sports_skill'].tolist()
departments = df['department'].tolist()
unique_depts = list(dict.fromkeys(departments)) # รายชื่อแผนกทั้งหมด

num_people = len(df)
half_team = num_people // 2  # โควตาสีละ 20 คน

# -----------------------------------------------------------
# 2. ฟังก์ชัน Fitness (ความเหมาะสม ยิ่งเข้าใกล้ 100 ยิ่งดี)
# -----------------------------------------------------------
def calculate_fitness(chromosome):
    """
    chromosome คือ list เช่น [0, 1, 0, 1, ...]
    0 = ทีมสีแดง, 1 = ทีมสีเขียว
    """
    # 1. คำนวณความแฟนเนจด้านกีฬา (Sports Fairness)
    red_skills = [skills[i] for i in range(num_people) if chromosome[i] == 0]
    green_skills = [skills[i] for i in range(num_people) if chromosome[i] == 1]
    
    avg_red = sum(red_skills) / len(red_skills)
    avg_green = sum(green_skills) / len(green_skills)
    sports_diff = abs(avg_red - avg_green)  # ผลต่างคะแนนเฉลี่ยกีฬา
    
    # 2. คำนวณการกระจายข้ามแผนก (Harmonization)
    dept_diff = 0
    for dept in unique_depts:
        count_red = sum(1 for i in range(num_people) if chromosome[i] == 0 and departments[i] == dept)
        count_green = sum(1 for i in range(num_people) if chromosome[i] == 1 and departments[i] == dept)
        dept_diff += abs(count_red - count_green)  # ผลต่างการแบ่งคนในแผนก
        
    # คิดคะแนนความเหมาะสมเต็ม 100 (หักลบผลต่างออกไป ยิ่งผลต่างน้อย คะแนนยิ่งสูง)
    fitness = 100 - (sports_diff * 3) - (dept_diff * 2)
    return fitness, avg_red, avg_green, sports_diff, dept_diff

# -----------------------------------------------------------
# 3. การทำงานของ Genetic Algorithm (GA)
# -----------------------------------------------------------
POP_SIZE = 60        # จำนวนโครโมโซมในแต่ละรุ่น
GENERATIONS = 100    # จำนวนรุ่นที่ให้วิวัฒนาการ
MUTATION_RATE = 0.20 # โอกาสเกิดการสลับตัวผู้เล่น (20%)

# สุ่มประชากรรุ่นแรก โดยให้มีสีแดง 20 คน และสีเขียว 20 คนเท่ากันพอดี
population = []
for _ in range(POP_SIZE):
    chrom = [0] * half_team + [1] * half_team
    random.shuffle(chrom)
    population.append(chrom)

best_fitness_history = []
best_chromosome = None
best_score = -999
best_info = None

print("กำลังค้นหาการจัดทีมที่ดีที่สุดด้วย Genetic Algorithm...")

for gen in range(1, GENERATIONS + 1):
    # ประเมินคะแนน Fitness ของทุกคนในรุ่น
    eval_results = [calculate_fitness(chrom) for chrom in population]
    fitness_list = [res[0] for res in eval_results]
    
    # บันทึกโครโมโซมที่ทำคะแนนได้ดีที่สุด
    max_idx = fitness_list.index(max(fitness_list))
    if fitness_list[max_idx] > best_score:
        best_score = fitness_list[max_idx]
        best_chromosome = population[max_idx].copy()
        best_info = eval_results[max_idx]
        
    best_fitness_history.append(best_score)
    
    # Elitism: นำ 2 ตัวที่ดีที่สุดไปไว้ในรุ่นถัดไปทันที
    sorted_indices = sorted(range(POP_SIZE), key=lambda k: fitness_list[k], reverse=True)
    new_population = [population[sorted_indices[0]].copy(), population[sorted_indices[1]].copy()]
    
    while len(new_population) < POP_SIZE:
        # Selection: สุ่มแข่งแบบ Tournament 2 ตัว ตัวไหนคะแนนดีกว่าชนะ
        i1, i2 = random.sample(range(POP_SIZE), 2)
        parent1 = population[i1] if fitness_list[i1] > fitness_list[i2] else population[i2]
        
        i3, i4 = random.sample(range(POP_SIZE), 2)
        parent2 = population[i3] if fitness_list[i3] > fitness_list[i4] else population[i4]
        
        # Crossover: สลับท่อนยีนระหว่างพ่อกับแม่
        cut = random.randint(1, num_people - 1)
        child = parent1[:cut] + parent2[cut:]
        
        # ซ่อมแซมให้ลูกมีจำนวนคนสีละ 20 คนเท่าเดิม
        ones_count = child.count(1)
        if ones_count > half_team:
            ones_pos = [i for i in range(num_people) if child[i] == 1]
            for pos in random.sample(ones_pos, ones_count - half_team):
                child[pos] = 0
        elif ones_count < half_team:
            zeros_pos = [i for i in range(num_people) if child[i] == 0]
            for pos in random.sample(zeros_pos, half_team - ones_count):
                child[pos] = 1
                
        # Mutation: สุ่มสลับตัวผู้เล่นสีแดงกับสีเขียว 1 คู่
        if random.random() < MUTATION_RATE:
            pos_red = random.choice([i for i in range(num_people) if child[i] == 0])
            pos_green = random.choice([i for i in range(num_people) if child[i] == 1])
            child[pos_red], child[pos_green] = 1, 0
            
        new_population.append(child)
        
    population = new_population

# -----------------------------------------------------------
# 4. สรุปผลลัพธ์
# -----------------------------------------------------------
fitness, avg_red, avg_green, sports_diff, dept_diff = best_info

print("=" * 60)
print(f" ผลลัพธ์ที่ดีที่สุด (Best Fitness: {fitness:.2f} / 100)")
print("=" * 60)
print(" 1. ความแฟนเนจด้านกีฬา (Sports Fairness):")
print(f"    - ค่าเฉลี่ยสกิล ทีมสีแดง  : {avg_red:.2f}")
print(f"    - ค่าเฉลี่ยสกิล ทีมสีเขียว: {avg_green:.2f}")
print(f"    - ผลต่างคะแนนเฉลี่ย      : {sports_diff:.2f} แต้ม (สูสีมาก!)")
print("-" * 60)
print(" 2. การกระจายตัวข้ามแผนก (Harmonization):")
for dept in unique_depts:
    c_red = sum(1 for i in range(num_people) if best_chromosome[i] == 0 and departments[i] == dept)
    c_green = sum(1 for i in range(num_people) if best_chromosome[i] == 1 and departments[i] == dept)
    print(f"    - แผนก {dept:<16}: สีแดง {c_red} คน | สีเขียว {c_green} คน")
print("=" * 60)

# บันทึกรายชื่อสมาชิกลงไฟล์ CSV
df['Team'] = ['สีแดง (Red)' if g == 0 else 'สีเขียว (Green)' for g in best_chromosome]
df.to_csv("outputs/team_assignment.csv", index=False, encoding="utf-8-sig")
print(">> บันทึกรายชื่อสมาชิกลงไฟล์: outputs/team_assignment.csv")

# วาดกราฟความฟิตเนสอย่างง่าย
plt.figure(figsize=(7, 4))
plt.plot(range(1, GENERATIONS + 1), best_fitness_history, color='blue', linewidth=2)
plt.title("Genetic Algorithm: Fitness Score over Generations")
plt.xlabel("Generation")
plt.ylabel("Fitness Score (0 - 100)")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/fitness_evolution.png", dpi=150)
print(">> บันทึกกราฟลงไฟล์: outputs/fitness_evolution.png")
