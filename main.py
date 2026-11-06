import random
import pandas as pd
import matplotlib.pyplot as plt

# 1. โหลดข้อมูลสมาชิกจากไฟล์ HR(in).csv (358 คน)
df = pd.read_csv("data/HR(in).csv")
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)  # ดัชนีมวลกาย

ages = df['age'].tolist()
bmis = df['bmi'].tolist()
genders = df['gender'].tolist()
depts = df['department'].tolist()

N, half = len(df), len(df) // 2  # N = 358 คน, ทีมละ 179 คนพอดี
unique_depts = sorted(set(depts))

# 2. ฟังก์ชัน Fitness (วัดความเหมาะสมของทีม)
def fitness(chrom):
    # 1) ด้านกีฬา (Sports Fairness): สัดส่วนเพศ, อายุเฉลี่ย, และ BMI เฉลี่ย ต้องใกล้เคียงกัน
    r_age = sum(ages[i] for i in range(N) if chrom[i] == 0) / half
    g_age = sum(ages[i] for i in range(N) if chrom[i] == 1) / half
    r_bmi = sum(bmis[i] for i in range(N) if chrom[i] == 0) / half
    g_bmi = sum(bmis[i] for i in range(N) if chrom[i] == 1) / half
    
    r_male = sum(1 for i in range(N) if chrom[i] == 0 and genders[i] == 'Male')
    g_male = sum(1 for i in range(N) if chrom[i] == 1 and genders[i] == 'Male')
    
    diff_age = abs(r_age - g_age)
    diff_bmi = abs(r_bmi - g_bmi)
    diff_male = abs(r_male - g_male)
    
    # 2) ด้านแผนก (Harmonization): ผลต่างการแบ่งคนของแต่ละแผนก (19 แผนก)
    diff_dept = 0
    for d in unique_depts:
        c_red = sum(1 for i in range(N) if chrom[i] == 0 and depts[i] == d)
        c_grn = sum(1 for i in range(N) if chrom[i] == 1 and depts[i] == d)
        diff_dept += abs(c_red - c_grn)
        
    score = 100 - (diff_age * 5) - (diff_bmi * 10) - (diff_male * 2) - (diff_dept * 1.2)
    return score, diff_age, diff_bmi, diff_male, diff_dept, r_age, g_age, r_bmi, g_bmi, r_male, g_male

# 3. Genetic Algorithm (GA)
POP_SIZE, GENS = 60, 120
pop = [[0] * half + [1] * half for _ in range(POP_SIZE)]
for c in pop: random.shuffle(c)

best_chrom, best_score, best_stats, history = None, -999, None, []

print("กำลังค้นหาการจัดทีมที่ดีที่สุดจากไฟล์ HR(in).csv ด้วย Genetic Algorithm...")

for _ in range(GENS):
    scores = [fitness(c) for c in pop]
    fits = [s[0] for s in scores]
    
    if max(fits) > best_score:
        best_score = max(fits)
        best_chrom = pop[fits.index(best_score)].copy()
        best_stats = scores[fits.index(best_score)]
    history.append(best_score)

    # Elitism: เก็บตัวที่ดีที่สุดไว้ 1 ตัว
    new_pop = [best_chrom.copy()]
    
    while len(new_pop) < POP_SIZE:
        # Selection: Tournament สุ่ม 2 ตัวแล้วเลือกตัวที่คะแนนดีกว่า
        p1 = pop[max(random.sample(range(POP_SIZE), 2), key=lambda x: fits[x])]
        p2 = pop[max(random.sample(range(POP_SIZE), 2), key=lambda x: fits[x])]

        # Crossover: สลับท่อนยีน
        cut = random.randint(1, N - 1)
        child = p1[:cut] + p2[cut:]

        # Repair: ปรับให้มีสีละ 179 คนเท่าเดิม
        while child.count(1) > half: child[random.choice([i for i, v in enumerate(child) if v == 1])] = 0
        while child.count(1) < half: child[random.choice([i for i, v in enumerate(child) if v == 0])] = 1

        # Mutation: สุ่มสลับตัวผู้เล่นสีแดงกับสีเขียว 1 คู่ (Swap)
        if random.random() < 0.25:
            i0 = random.choice([i for i, v in enumerate(child) if v == 0])
            i1 = random.choice([i for i, v in enumerate(child) if v == 1])
            child[i0], child[i1] = 1, 0

        new_pop.append(child)
    pop = new_pop

# 4. แสดงผลลัพธ์ใน Terminal
score, d_age, d_bmi, d_male, d_dept, r_age, g_age, r_bmi, g_bmi, r_male, g_male = best_stats

print("=" * 60)
print(f"ผลลัพธ์ที่ดีที่สุด (Fitness: {score:.2f} / 100)")
print("=" * 60)
print(f" 1. ความแฟนเนจด้านกีฬา (Sports Fairness):")
print(f"    - สัดส่วนผู้ชาย (Male) : สีแดง {r_male} คน | สีเขียว {g_male} คน (ต่าง {d_male} คน)")
print(f"    - ค่าเฉลี่ยอายุ (Age)  : สีแดง {r_age:.2f} ปี | สีเขียว {g_age:.2f} ปี (ต่าง {d_age:.2f} ปี)")
print(f"    - ค่าเฉลี่ยดัชนีมวลกาย : สีแดง {r_bmi:.2f} | สีเขียว {g_bmi:.2f} (ต่าง {d_bmi:.2f})")
print("-" * 60)
print(f" 2. การกระจายตัวข้ามแผนก (Harmonization) ทั้ง 19 แผนก:")
for d in unique_depts:
    cr = sum(1 for i in range(N) if best_chrom[i] == 0 and depts[i] == d)
    cg = sum(1 for i in range(N) if best_chrom[i] == 1 and depts[i] == d)
    print(f"    - แผนก {d:<2}: สีแดง {cr:<2} คน | สีเขียว {cg:<2} คน")
print("=" * 60)

# 5. บันทึกผลลง CSV
df['Team'] = ['สีแดง (Red)' if v == 0 else 'สีเขียว (Green)' for v in best_chrom]
df.to_csv("outputs/team_assignment.csv", index=False, encoding="utf-8-sig")
print(">> บันทึกรายชื่อผลลัพธ์ลง: outputs/team_assignment.csv")

# -------------------------------------------------------------
# 6. รวมกราฟทั้ง 3 ด้านไว้ในไฟล์เดียว (3-in-1 Combined Dashboard)
# -------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['Leelawadee UI', 'Tahoma', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(15, 8.5))

# กราฟ 1 (บนซ้าย): การพัฒนาของค่า Fitness (Fitness Progression)
ax1 = plt.subplot(2, 2, 1)
ax1.plot(history, color='#1976D2', linewidth=2)
ax1.set_title("1. Genetic Algorithm: Fitness Progression", fontsize=12, fontweight='bold')
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness Score")
ax1.grid(True, linestyle='--', alpha=0.6)

# กราฟ 2 (บนขวา): ความสูสีทางกายภาพ/กีฬา (Sports Fairness Comparison)
ax2 = plt.subplot(2, 2, 2)
metric_labels = ["Male (Count)", "Mean Age (Yrs)", "Mean BMI"]
red_metrics = [r_male, r_age, r_bmi]
grn_metrics = [g_male, g_age, g_bmi]
x_met = range(len(metric_labels))
b1 = ax2.bar([i - 0.2 for i in x_met], red_metrics, width=0.4, label="Red", color="#E53935", edgecolor="black", alpha=0.85)
b2 = ax2.bar([i + 0.2 for i in x_met], grn_metrics, width=0.4, label="Green", color="#43A047", edgecolor="black", alpha=0.85)
ax2.set_xticks(list(x_met))
ax2.set_xticklabels(metric_labels, fontsize=10)
ax2.set_title("2. Sports Fairness Comparison (Physical Metrics)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Value")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

# แสดงตัวเลขบนแท่งกราฟด้านกีฬา
for b in b1 + b2:
    val = b.get_height()
    ax2.annotate(f"{val:.1f}" if val < 50 else f"{int(val)}", 
                 xy=(b.get_x() + b.get_width() / 2, val),
                 xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# กราฟ 3 (ล่างยาว): การกระจายตัวข้ามแผนก (Cross-Department Harmonization)
ax3 = plt.subplot(2, 1, 2)
dept_r = [sum(1 for i in range(N) if best_chrom[i] == 0 and depts[i] == d) for d in unique_depts]
dept_g = [sum(1 for i in range(N) if best_chrom[i] == 1 and depts[i] == d) for d in unique_depts]
x_dept = range(len(unique_depts))
bd1 = ax3.bar([i - 0.2 for i in x_dept], dept_r, width=0.4, label="Red", color="#E53935", edgecolor="black", alpha=0.85)
bd2 = ax3.bar([i + 0.2 for i in x_dept], dept_g, width=0.4, label="Green", color="#43A047", edgecolor="black", alpha=0.85)
ax3.set_xticks(list(x_dept))
ax3.set_xticklabels(unique_depts, fontsize=9)
ax3.set_title("3. Cross-Department Harmonization: Distribution across Teams (19 Departments)", fontsize=12, fontweight='bold')
ax3.set_xlabel("Department")
ax3.set_ylabel("Number of Members")
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.6)

# แสดงตัวเลขบนแท่งกราฟแผนก
for b in bd1 + bd2:
    val = b.get_height()
    if val > 0:
        ax3.annotate(f"{int(val)}", 
                     xy=(b.get_x() + b.get_width() / 2, val),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig("outputs/summary_plots.png", dpi=150)
plt.savefig("outputs/fitness_evolution.png", dpi=150)  # ซิงค์ชื่อนี้ด้วยเพื่อให้ README บน GitHub แสดงรูปภาพรวมทันที
plt.close()
print(">> รวมกราฟทั้ง 3 รูปไว้ในไฟล์เดียว: outputs/summary_plots.png เรียบร้อยแล้ว!")
