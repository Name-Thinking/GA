"""
=================================================================================
 🏆 GeneTeam: สคริปต์ฉบับเข้าใจง่ายที่สุด (Simple & Clean Single File)
=================================================================================
 สคริปต์นี้รวมการทำงานทั้งหมดของ Genetic Algorithm ไว้ในไฟล์เดียว
 เพื่อให้อ่าน ทำความเข้าใจ และนำไปอธิบายให้อาจารย์ฟังได้อย่างง่ายดาย!

 🧬 องค์ประกอบหลักของ GA:
   1. โครโมโซม (Chromosome): อาร์เรย์ของเลข 0 (สีแดง) และ 1 (สีเขียว)
   2. ฟิตเนส (Fitness): วัด 2 อย่าง คือ (สกิลกีฬาสูสีกัน) + (กระจายแผนกเท่ากัน)
   3. การคัดเลือก (Selection): สุ่มแข่งเป็นกลุ่ม (Tournament) ตัวเก่งได้สิทธิ์ผสมพันธุ์
   4. การผสมพันธุ์ (Crossover): แลกเปลี่ยนท่อนยีนระหว่างพ่อกับแม่
   5. การกลายพันธุ์ (Mutation): สลับตัวผู้เล่นสีแดงกับสีเขียว (Swap)
=================================================================================
"""

import pandas as pd
import numpy as np

# -------------------------------------------------------------------------------
# 1. โหลดข้อมูลสมาชิก (Data Loading)
# -------------------------------------------------------------------------------
print(">> ขั้นตอนที่ 1: กำลังโหลดข้อมูลสมาชิก...")
df = pd.read_csv("data/company_members.csv")

names = df['name'].values
depts = df['department'].values
skills = df['sports_skill'].values.astype(float)
total_people = len(df)
unique_depts, dept_ids = np.unique(depts, return_inverse=True)

print(f"   โหลดข้อมูลสำเร็จ: ทั้งหมด {total_people} คน จาก {len(unique_depts)} แผนก")

# -------------------------------------------------------------------------------
# 2. ฟังก์ชันคำนวณความเหมาะสม (Fitness Function)
# -------------------------------------------------------------------------------
def calculate_fitness(chromosome):
    """
    รับ chromosome เช่น [0, 1, 1, 0, ...]
    คืนค่าคะแนนความเหมาะสม (เต็ม 100) ยิ่งเข้าใกล้ 100 ยิ่งดีเลิศ
    """
    red_indices = (chromosome == 0)
    green_indices = (chromosome == 1)

    # 1. ด้านกีฬา (Sports Fairness)
    # หาค่าเฉลี่ยสกิลกีฬาของแต่ละสี
    red_skill_avg = np.mean(skills[red_indices])
    green_skill_avg = np.mean(skills[green_indices])
    skill_diff = abs(red_skill_avg - green_skill_avg)

    # คะแนนกีฬา: ถ้าผลต่าง = 0 จะได้ 100 เต็ม
    sports_score = max(0.0, 100.0 - (skill_diff * 3.0))

    # 2. ด้านแผนก (Cross-Department Harmonization)
    # ดูว่าแต่ละแผนกถูกแบ่งครึ่งลง 2 สีเท่ากันไหม
    dept_counts_red = np.bincount(dept_ids[red_indices], minlength=len(unique_depts))
    dept_counts_green = np.bincount(dept_ids[green_indices], minlength=len(unique_depts))

    # นับว่ามีแผนกไหนที่คนเอียงไปข้างใดข้างหนึ่งมากเกินไป
    imbalance = 0
    for i in range(len(unique_depts)):
        diff = abs(dept_counts_red[i] - dept_counts_green[i])
        # ถ้าคนในแผนกเป็นเลขคี่ ยอมให้ต่างกันได้ 1 คน
        total_in_dept = dept_counts_red[i] + dept_counts_green[i]
        allowed = total_in_dept % 2
        imbalance += max(0, diff - allowed)

    harmonize_score = max(0.0, 100.0 - (imbalance / total_people * 100.0))

    # รวมคะแนนทั้ง 2 ด้านอย่างละครึ่ง (น้ำหนัก 50:50)
    total_fitness = (0.50 * sports_score) + (0.50 * harmonize_score)
    return total_fitness, sports_score, harmonize_score, red_skill_avg, green_skill_avg

# -------------------------------------------------------------------------------
# 3. กลไก Genetic Algorithm (GA Engine)
# -------------------------------------------------------------------------------
POP_SIZE = 80       # จำนวนประชากรในแต่ละรุ่น
GENERATIONS = 120   # จำนวนรุ่นที่ให้วิวัฒนาการ
MUTATION_RATE = 0.20 # โอกาสเกิดการกลายพันธุ์

# สร้างประชากรรุ่นแรก โดยให้มีสีแดง 20 คน และสีเขียว 20 คนเท่ากันพอดี
half = total_people // 2
population = []
for _ in range(POP_SIZE):
    chrom = np.array([0] * half + [1] * (total_people - half))
    np.random.shuffle(chrom)
    population.append(chrom)

print(">> ขั้นตอนที่ 2: เริ่มต้นการวิวัฒนาการด้วย Genetic Algorithm...")

best_chromosome = None
best_fitness = -1.0
best_stats = None

for gen in range(1, GENERATIONS + 1):
    # ประเมินคะแนนประชากรทุกคนในรุ่นนี้
    results = [calculate_fitness(chrom) for chrom in population]
    fitnesses = [res[0] for res in results]

    # บันทึกโครโมโซมที่เก่งที่สุด
    current_best_idx = np.argmax(fitnesses)
    if fitnesses[current_best_idx] > best_fitness:
        best_fitness = fitnesses[current_best_idx]
        best_chromosome = population[current_best_idx].copy()
        best_stats = results[current_best_idx]

    # สร้างประชากรรุ่นถัดไป (Elitism: เก็บตัวที่ดีที่สุดไว้ 2 ตัวเสมอ)
    sorted_idx = np.argsort(fitnesses)[::-1]
    new_population = [population[sorted_idx[0]].copy(), population[sorted_idx[1]].copy()]

    while len(new_population) < POP_SIZE:
        # Selection: สุ่มแข่งแบบ Tournament
        cand1, cand2 = np.random.choice(POP_SIZE, 2, replace=False)
        p1 = population[cand1] if fitnesses[cand1] > fitnesses[cand2] else population[cand2]

        cand3, cand4 = np.random.choice(POP_SIZE, 2, replace=False)
        p2 = population[cand3] if fitnesses[cand3] > fitnesses[cand4] else population[cand4]

        # Crossover: สลับท่อนยีน
        child = p1.copy()
        cut = np.random.randint(1, total_people)
        child[cut:] = p2[cut:]

        # ซ่อมแซมลูกให้มีจำนวน 0 และ 1 เท่ากัน 50:50
        ones = np.where(child == 1)[0]
        if len(ones) > half:
            child[np.random.choice(ones, len(ones) - half, replace=False)] = 0
        elif len(ones) < half:
            zeros = np.where(child == 0)[0]
            child[np.random.choice(zeros, half - len(ones), replace=False)] = 1

        # Mutation: สลับตัวผู้เล่นสีแดงกับสีเขียว 1 คู่
        if np.random.rand() < MUTATION_RATE:
            z = np.random.choice(np.where(child == 0)[0])
            o = np.random.choice(np.where(child == 1)[0])
            child[z], child[o] = 1, 0

        new_population.append(child)

    population = new_population

# -------------------------------------------------------------------------------
# 4. แสดงผลลัพธ์ที่ได้ (Print Results)
# -------------------------------------------------------------------------------
tot_fit, sp_score, hm_score, red_avg, grn_avg = best_stats

print("-" * 65)
print(" 🎉 การจัดทีมเสร็จสมบูรณ์! (Optimization Finished)")
print("-" * 65)
print(f" คะแนนรวมความฟิตเนส (Total Fitness) : {tot_fit:.2f} / 100")
print(f" คะแนนความแฟนเนจด้านกีฬา (Sports)   : {sp_score:.2f} / 100")
print(f" คะแนนความกลมเกลียวข้ามแผนก (Harmony): {hm_score:.2f} / 100")
print(f" ค่าเฉลี่ยสกิลกีฬา ทีมสีแดง          : {red_avg:.2f}")
print(f" ค่าเฉลี่ยสกิลกีฬา ทีมสีเขียว        : {grn_avg:.2f}")
print(f" ผลต่างสกิลเฉลี่ยระหว่าง 2 สี        : {abs(red_avg - grn_avg):.2f} แต้ม (สูสีมาก!)")
print("-" * 65)

# ส่งออกผลลัพธ์เป็นไฟล์ CSV
output_df = df.copy()
output_df['Team'] = np.where(best_chromosome == 0, 'สีแดง (Red)', 'สีเขียว (Green)')
output_df.to_csv('outputs/team_assignment_simple.csv', index=False, encoding='utf-8-sig')
print(">> บันทึกรายชื่อผลลัพธ์ลงไฟล์: outputs/team_assignment_simple.csv เรียบร้อย")
