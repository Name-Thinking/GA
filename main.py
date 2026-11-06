import random
import pandas as pd
import matplotlib.pyplot as plt

# 1. โหลดข้อมูลสมาชิก (40 คน)
df = pd.read_csv("data/company_members.csv")
skills = df['sports_skill'].tolist()
depts = df['department'].tolist()
N, half = len(df), len(df) // 2  # N = 40, half = 20

# 2. ฟังก์ชัน Fitness (ประเมินความเหมาะสม)
def fitness(chrom):
    # 1) ด้านกีฬา: ผลต่างค่าเฉลี่ยสกิล (ยิ่งน้อยยิ่งสูสี)
    red_s = [skills[i] for i in range(N) if chrom[i] == 0]
    grn_s = [skills[i] for i in range(N) if chrom[i] == 1]
    diff_sport = abs(sum(red_s)/half - sum(grn_s)/half)
    
    # 2) ด้านแผนก: ผลต่างการแบ่งคนแต่ละแผนก (ยิ่งน้อยยิ่งกระจายดี)
    diff_dept = sum(abs(sum(1 for i in range(N) if chrom[i] == 0 and depts[i] == d) -
                        sum(1 for i in range(N) if chrom[i] == 1 and depts[i] == d))
                    for d in set(depts))
    
    score = 100 - (diff_sport * 3) - (diff_dept * 2)
    return score, diff_sport, diff_dept

# 3. Genetic Algorithm (GA)
POP_SIZE, GENS = 50, 80
pop = [[0] * half + [1] * half for _ in range(POP_SIZE)]
for c in pop: random.shuffle(c)

best_chrom, best_score, history = None, -999, []

for _ in range(GENS):
    scores = [fitness(c) for c in pop]
    fits = [s[0] for s in scores]
    
    if max(fits) > best_score:
        best_score = max(fits)
        best_chrom = pop[fits.index(best_score)].copy()
    history.append(best_score)

    # Elitism: เก็บตัวที่ดีที่สุดไว้ 1 ตัว
    new_pop = [best_chrom.copy()]
    
    while len(new_pop) < POP_SIZE:
        # Selection: Tournament สุ่ม 2 ตัวแล้วเลือกตัวที่ดีกว่า
        p1 = pop[max(random.sample(range(POP_SIZE), 2), key=lambda x: fits[x])]
        p2 = pop[max(random.sample(range(POP_SIZE), 2), key=lambda x: fits[x])]

        # Crossover: สลับท่อนยีน
        cut = random.randint(1, N - 1)
        child = p1[:cut] + p2[cut:]

        # Repair: ปรับให้มีสีละ 20 คนเท่าเดิม
        while child.count(1) > half: child[random.choice([i for i, v in enumerate(child) if v == 1])] = 0
        while child.count(1) < half: child[random.choice([i for i, v in enumerate(child) if v == 0])] = 1

        # Mutation: สุ่มสลับตำแหน่งคนสีแดงกับสีเขียว (Swap)
        if random.random() < 0.2:
            i0 = random.choice([i for i, v in enumerate(child) if v == 0])
            i1 = random.choice([i for i, v in enumerate(child) if v == 1])
            child[i0], child[i1] = 1, 0

        new_pop.append(child)
    pop = new_pop

# 4. สรุปผลลัพธ์
score, diff_s, diff_d = fitness(best_chrom)
print("=" * 55)
print(f"ผลลัพธ์ที่ดีที่สุด (Fitness: {score:.2f} / 100)")
print(f"- ผลต่างคะแนนเฉลี่ยกีฬา : {diff_s:.2f} แต้ม (สูสีมาก)")
print(f"- ผลต่างการกระจายแผนก  : {diff_d} คน")
print("-" * 55)
for d in sorted(set(depts)):
    r = sum(1 for i in range(N) if best_chrom[i] == 0 and depts[i] == d)
    g = sum(1 for i in range(N) if best_chrom[i] == 1 and depts[i] == d)
    print(f"แผนก {d:<16}: สีแดง {r} คน | สีเขียว {g} คน")
print("=" * 55)

# 5. บันทึกผลลง CSV และสร้างกราฟ
df['Team'] = ['สีแดง (Red)' if v == 0 else 'สีเขียว (Green)' for v in best_chrom]
df.to_csv("outputs/team_assignment.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(6, 3.5))
plt.plot(history, color='blue', linewidth=2)
plt.title("Fitness Evolution")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.tight_layout()
plt.savefig("outputs/fitness_evolution.png", dpi=120)
plt.close()
