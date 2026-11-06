import os
import random
import pandas as pd
import matplotlib.pyplot as plt

# ล็อก Seed เพื่อให้ผลลัพธ์แม่นยำและตรงกันทุกครั้งที่รัน
random.seed(42)

# 1. โหลดข้อมูลสมาชิกจากไฟล์ HR(in).csv (358 คน)
df = pd.read_csv("data/HR(in).csv")
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

ages = df['age'].tolist()
weights = df['weight'].tolist()
heights = df['height'].tolist()
bmis = df['bmi'].tolist()
genders = df['gender'].tolist()
depts = df['department'].tolist()

N, half = len(df), len(df) // 2  # N = 358 คน (แบ่งทีมละ 179 คนเท่ากัน)
unique_depts = sorted(set(depts))
dept_totals = {d: depts.count(d) for d in unique_depts}

# 2. ฟังก์ชัน Fitness (วัดความเหมาะสม เต็ม 100 คะแนน)
def fitness(chrom):
    # 1) ด้านกีฬา (Sports Fairness): เพศ, อายุ, น้ำหนัก, ส่วนสูง, BMI ต้องสมดุลกัน
    r_age = sum(ages[i] for i in range(N) if chrom[i] == 0) / half
    g_age = sum(ages[i] for i in range(N) if chrom[i] == 1) / half
    r_w = sum(weights[i] for i in range(N) if chrom[i] == 0) / half
    g_w = sum(weights[i] for i in range(N) if chrom[i] == 1) / half
    r_h = sum(heights[i] for i in range(N) if chrom[i] == 0) / half
    g_h = sum(heights[i] for i in range(N) if chrom[i] == 1) / half
    r_bmi = sum(bmis[i] for i in range(N) if chrom[i] == 0) / half
    g_bmi = sum(bmis[i] for i in range(N) if chrom[i] == 1) / half
    
    r_male = sum(1 for i in range(N) if chrom[i] == 0 and genders[i] == 'Male')
    g_male = sum(1 for i in range(N) if chrom[i] == 1 and genders[i] == 'Male')
    
    diff_age = abs(r_age - g_age)
    diff_w = abs(r_w - g_w)
    diff_h = abs(r_h - g_h)
    diff_bmi = abs(r_bmi - g_bmi)
    diff_male = abs(r_male - g_male)
    
    # 2) ด้านแผนก (Harmonization): ผลต่างการแบ่งคนทั้ง 19 แผนก (หักเศษเลขคี่ที่แบ่งครึ่งไม่ได้)
    excess_dept_diff = 0
    for d in unique_depts:
        c_red = sum(1 for i in range(N) if chrom[i] == 0 and depts[i] == d)
        c_grn = dept_totals[d] - c_red
        excess_dept_diff += abs(c_red - c_grn) - (dept_totals[d] % 2)
        
    score = 100 - (diff_age * 4) - (diff_w * 3) - (diff_h * 3) - (diff_bmi * 5) - (diff_male * 2) - (excess_dept_diff * 1.5)
    return (score, diff_age, diff_w, diff_h, diff_bmi, diff_male, excess_dept_diff,
            r_age, g_age, r_w, g_w, r_h, g_h, r_bmi, g_bmi, r_male, g_male)

# 3. Genetic Algorithm (GA)
POP_SIZE, GENS = 60, 120
pop = [[0] * half + [1] * half for _ in range(POP_SIZE)]
for c in pop: random.shuffle(c)

best_chrom, best_score, best_stats = None, -999, None
best_history, avg_history = [], []

print("กำลังประมวลผลจัดทีมด้วย Genetic Algorithm จากไฟล์ HR(in).csv...")

for _ in range(GENS):
    scores = [fitness(c) for c in pop]
    fits = [s[0] for s in scores]
    
    if max(fits) > best_score:
        best_score = max(fits)
        best_chrom = pop[fits.index(best_score)].copy()
        best_stats = scores[fits.index(best_score)]
        
    best_history.append(best_score)
    avg_history.append(sum(fits) / POP_SIZE)

    # Elitism: เก็บคำตอบที่ดีที่สุดไว้เสมอ
    new_pop = [best_chrom.copy()]
    
    while len(new_pop) < POP_SIZE:
        # Selection: Tournament
        p1 = pop[max(random.sample(range(POP_SIZE), 2), key=lambda x: fits[x])]
        p2 = pop[max(random.sample(range(POP_SIZE), 2), key=lambda x: fits[x])]

        # Crossover: สลับท่อนยีน
        cut = random.randint(1, N - 1)
        child = p1[:cut] + p2[cut:]

        # Repair: ปรับให้มีสีละ 179 คนเท่ากันเป๊ะ
        while child.count(1) > half: child[random.choice([i for i, v in enumerate(child) if v == 1])] = 0
        while child.count(1) < half: child[random.choice([i for i, v in enumerate(child) if v == 0])] = 1

        # Mutation: สลับตำแหน่งคนสีแดงกับสีเขียว (Swap)
        if random.random() < 0.30:
            i0 = random.choice([i for i, v in enumerate(child) if v == 0])
            i1 = random.choice([i for i, v in enumerate(child) if v == 1])
            child[i0], child[i1] = 1, 0

        new_pop.append(child)
    pop = new_pop

# 4. แสดงผลลัพธ์ที่แม่นยำใน Terminal
(score, d_age, d_w, d_h, d_bmi, d_male, d_dept,
 r_age, g_age, r_w, g_w, r_h, g_h, r_bmi, g_bmi, r_male, g_male) = best_stats
r_female, g_female = half - r_male, half - g_male

print("=" * 65)
print(f" ผลลัพธ์การจัดทีมที่ดีที่สุด (Best Fitness: {score:.2f} / 100)")
print("=" * 65)
print(" 1. ความแฟนเนจด้านกีฬา (Sports Fairness - เปรียบเทียบสีแดง vs สีเขียว):")
print(f"    - จำนวนสมาชิกทั้งหมด : สีแดง {half} คน | สีเขียว {half} คน")
print(f"    - สัดส่วนผู้ชาย (Male) : สีแดง {r_male} คน | สีเขียว {g_male} คน (ต่าง {d_male} คน)")
print(f"    - สัดส่วนผู้หญิง (Female): สีแดง {r_female} คน | สีเขียว {g_female} คน (ต่าง {abs(r_female-g_female)} คน)")
print(f"    - น้ำหนักเฉลี่ย (Weight): สีแดง {r_w:.2f} kg | สีเขียว {g_w:.2f} kg (ต่าง {d_w:.2f} kg)")
print(f"    - ส่วนสูงเฉลี่ย (Height): สีแดง {r_h:.2f} cm | สีเขียว {g_h:.2f} cm (ต่าง {d_h:.2f} cm)")
print(f"    - อายุเฉลี่ย (Age)    : สีแดง {r_age:.2f} ปี | สีเขียว {g_age:.2f} ปี (ต่าง {d_age:.2f} ปี)")
print(f"    - ดัชนีมวลกาย (BMI)   : สีแดง {r_bmi:.2f} | สีเขียว {g_bmi:.2f} (ต่าง {d_bmi:.2f})")
print("-" * 65)
print(f" 2. การกระจายตัวข้ามแผนก (Harmonization - ครบทั้ง 19 แผนก):")
for d in unique_depts:
    cr = sum(1 for i in range(N) if best_chrom[i] == 0 and depts[i] == d)
    cg = sum(1 for i in range(N) if best_chrom[i] == 1 and depts[i] == d)
    print(f"    - แผนก {d:<2} (รวม {cr+cg:<2} คน): สีแดง {cr:<2} คน | สีเขียว {cg:<2} คน")
print("=" * 65)

# 5. บันทึกผลลง CSV
os.makedirs("outputs", exist_ok=True)
df['bmi'] = df['bmi'].round(2)
df['Team'] = ['สีแดง (Red)' if v == 0 else 'สีเขียว (Green)' for v in best_chrom]
df.to_csv("outputs/team_assignment.csv", index=False, encoding="utf-8-sig")
print(">> บันทึกรายชื่อผลลัพธ์ลง: outputs/team_assignment.csv")

# 6. สร้างกราฟรวม 3 ด้านในไฟล์เดียว และเปิดแสดงผลกราฟ
plt.rcParams['font.sans-serif'] = ['Leelawadee UI', 'Tahoma', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(16, 9))

# กราฟที่ 1: Fitness Progression
ax1 = plt.subplot(2, 2, 1)
ax1.plot(range(1, GENS + 1), best_history, color='#D32F2F', linewidth=2.5, label="Best Fitness")
ax1.plot(range(1, GENS + 1), avg_history, color='#1976D2', linestyle='--', linewidth=1.8, label="Mean Population Fitness")
ax1.set_title(f"1. Genetic Algorithm Convergence (Best: {score:.2f}/100)", fontsize=12, fontweight='bold')
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fitness Score")
ax1.legend(loc="lower right")
ax1.grid(True, linestyle='--', alpha=0.6)

# กราฟที่ 2: Sports Fairness Comparison (6 ตัวชี้วัดทางกายภาพ)
ax2 = plt.subplot(2, 2, 2)
metric_labels = ["Male\n(คน)", "Female\n(คน)", "Weight\n(kg)", "Height\n(cm)", "Age\n(ปี)", "BMI"]
red_metrics = [r_male, r_female, r_w, r_h, r_age, r_bmi]
grn_metrics = [g_male, g_female, g_w, g_h, g_age, g_bmi]
x_met = range(len(metric_labels))
b1 = ax2.bar([i - 0.2 for i in x_met], red_metrics, width=0.4, label="สีแดง (Red)", color="#E53935", edgecolor="black", alpha=0.85)
b2 = ax2.bar([i + 0.2 for i in x_met], grn_metrics, width=0.4, label="สีเขียว (Green)", color="#43A047", edgecolor="black", alpha=0.85)
ax2.set_xticks(list(x_met))
ax2.set_xticklabels(metric_labels, fontsize=9)
ax2.set_ylim(0, max(red_metrics + grn_metrics) * 1.15)
ax2.set_title("2. Sports Fairness Comparison (ความสมดุลทางกายภาพ)", fontsize=12, fontweight='bold')
ax2.set_ylabel("Value")
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

for b in b1 + b2:
    val = b.get_height()
    txt = f"{int(val)}" if val in [r_male, g_male, r_female, g_female] else f"{val:.1f}"
    ax2.annotate(txt, xy=(b.get_x() + b.get_width() / 2, val),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

# กราฟที่ 3: Cross-Department Harmonization (19 แผนก)
ax3 = plt.subplot(2, 1, 2)
dept_r = [sum(1 for i in range(N) if best_chrom[i] == 0 and depts[i] == d) for d in unique_depts]
dept_g = [sum(1 for i in range(N) if best_chrom[i] == 1 and depts[i] == d) for d in unique_depts]
x_dept = range(len(unique_depts))
bd1 = ax3.bar([i - 0.2 for i in x_dept], dept_r, width=0.4, label="สีแดง (Red)", color="#E53935", edgecolor="black", alpha=0.85)
bd2 = ax3.bar([i + 0.2 for i in x_dept], dept_g, width=0.4, label="สีเขียว (Green)", color="#43A047", edgecolor="black", alpha=0.85)
ax3.set_xticks(list(x_dept))
ax3.set_xticklabels([f"Dept {d}\n({dept_totals[d]})" for d in unique_depts], fontsize=8.5)
ax3.set_ylim(0, max(dept_r + dept_g) * 1.15)
ax3.set_title("3. Cross-Department Harmonization: การกระจายสมาชิกทั้ง 19 แผนก (A - S)", fontsize=12, fontweight='bold')
ax3.set_xlabel("Department (จำนวนคนรวม)")
ax3.set_ylabel("Number of Members")
ax3.legend()
ax3.grid(True, linestyle='--', alpha=0.6)

for b in bd1 + bd2:
    val = b.get_height()
    if val > 0:
        ax3.annotate(f"{int(val)}", xy=(b.get_x() + b.get_width() / 2, val),
                     xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontsize=8, fontweight='bold')

plt.tight_layout()
plot_path = os.path.abspath("outputs/summary_plots.png")
plt.savefig(plot_path, dpi=150)
plt.savefig("outputs/fitness_evolution.png", dpi=150)
print(f">> บันทึกกราฟสำเร็จ: {plot_path}")

# เปิดแสดงรูปกราฟขึ้นมาบนหน้าจออัตโนมัติ
try:
    os.startfile(plot_path)
except Exception:
    pass
plt.close()
