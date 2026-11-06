import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# -------------------------------------------------------------
# กำหนดค่า Seed เพื่อให้การสุ่มคงที่และตรวจสอบผลซ้ำได้แม่นยำ
# -------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# -------------------------------------------------------------
# 1. โหลดข้อมูลสมาชิกจากไฟล์ HR(in).csv (358 คน)
# -------------------------------------------------------------
df = pd.read_csv("data/HR(in).csv")
df['department'] = df['department'].str.strip()
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

# คำนวณดัชนีความพร้อมด้านกีฬา (Sports Skill / Capability Index: 0 - 100)
# ประเมินจากความสมดุลของร่างกาย (BMI ใกล้เคียงค่ามาตรฐาน 22, อายุ และสัดส่วนความสูง)
bmi_score = 100 - np.clip(np.abs(df['bmi'] - 22.0) * 4.0, 0, 50)
age_score = 100 - np.clip((df['age'] - 20) * 1.5, 0, 50)
height_score = np.clip((df['height'] - 140) / (190 - 140) * 100, 30, 100)
df['sports_skill'] = (0.45 * bmi_score + 0.35 * age_score + 0.20 * height_score).round(1)

skills = df['sports_skill'].tolist()
ages = df['age'].tolist()
weights = df['weight'].tolist()
heights = df['height'].tolist()
bmis = df['bmi'].tolist()
genders = df['gender'].tolist()
depts = df['department'].tolist()

N = len(df)
unique_depts = sorted(set(depts))
dept_totals = {d: depts.count(d) for d in unique_depts}

# -------------------------------------------------------------
# 2. ฟังก์ชัน Fitness (ประเมินความเหมาะสมตามหลัก Genetic Algorithm)
#    * ไม่มีการใช้ลูปซ่อมแซมหรือรีบาลานซ์ (No Rebalance / Repair Loop)
#    * ให้ GA เรียนรู้และปรับสมดุลเองผ่านฟังก์ชันลงโทษ (Penalty Function)
# -------------------------------------------------------------
def fitness(chrom):
    r_count = sum(1 for v in chrom if v == 0)
    g_count = N - r_count
    
    # หากมีทีมใดไม่มีสมาชิกเลย ให้คะแนนต่ำสุด
    if r_count == 0 or g_count == 0:
        return -9999.0, None
    
    # 1) บทลงโทษกรณีขนาดทีมไม่เท่ากัน (Size Imbalance Penalty)
    size_diff = abs(r_count - g_count)
    size_penalty = size_diff * 8.0
    
    # 2) ความแฟนเนจด้านกีฬา (Sports Fairness)
    r_skill = sum(skills[i] for i in range(N) if chrom[i] == 0) / r_count
    g_skill = sum(skills[i] for i in range(N) if chrom[i] == 1) / g_count
    diff_skill = abs(r_skill - g_skill)
    
    r_male = sum(1 for i in range(N) if chrom[i] == 0 and genders[i] == 'Male')
    g_male = sum(1 for i in range(N) if chrom[i] == 1 and genders[i] == 'Male')
    diff_male = abs(r_male - g_male)
    
    r_age = sum(ages[i] for i in range(N) if chrom[i] == 0) / r_count
    g_age = sum(ages[i] for i in range(N) if chrom[i] == 1) / g_count
    diff_age = abs(r_age - g_age)
    
    r_w = sum(weights[i] for i in range(N) if chrom[i] == 0) / r_count
    g_w = sum(weights[i] for i in range(N) if chrom[i] == 1) / g_count
    diff_w = abs(r_w - g_w)
    
    r_h = sum(heights[i] for i in range(N) if chrom[i] == 0) / r_count
    g_h = sum(heights[i] for i in range(N) if chrom[i] == 1) / g_count
    diff_h = abs(r_h - g_h)
    
    r_bmi = sum(bmis[i] for i in range(N) if chrom[i] == 0) / r_count
    g_bmi = sum(bmis[i] for i in range(N) if chrom[i] == 1) / g_count
    diff_bmi = abs(r_bmi - g_bmi)
    
    # 3) การกระจายตัวข้ามแผนก (Department Harmonization)
    excess_dept_diff = 0
    for d in unique_depts:
        cr = sum(1 for i in range(N) if chrom[i] == 0 and depts[i] == d)
        cg = dept_totals[d] - cr
        # หักเศษคี่ที่แบ่งครึ่งเป๊ะไม่ได้ออก 1 คน
        excess_dept_diff += abs(cr - cg) - (dept_totals[d] % 2)
        
    score = 100.0 - size_penalty - (diff_skill * 5.0) - (diff_male * 2.0) - (diff_age * 2.0) - (diff_bmi * 3.0) - (excess_dept_diff * 1.5)
    
    stats = {
        'r_count': r_count, 'g_count': g_count, 'size_diff': size_diff,
        'r_skill': r_skill, 'g_skill': g_skill, 'diff_skill': diff_skill,
        'r_male': r_male, 'g_male': g_male, 'diff_male': diff_male,
        'r_age': r_age, 'g_age': g_age, 'diff_age': diff_age,
        'r_w': r_w, 'g_w': g_w, 'diff_w': diff_w,
        'r_h': r_h, 'g_h': g_h, 'diff_h': diff_h,
        'r_bmi': r_bmi, 'g_bmi': g_bmi, 'diff_bmi': diff_bmi,
        'dept_diff': excess_dept_diff
    }
    return score, stats

# -------------------------------------------------------------
# 3. การประมวลผลด้วย Genetic Algorithm (Pure GA - นำรีบาลานซ์ออก)
# -------------------------------------------------------------
POP_SIZE, GENS = 80, 150

# เริ่มต้นประชากรแบบสุ่มอิสระ (Pure Random Binary Initialization)
pop = [[random.randint(0, 1) for _ in range(N)] for _ in range(POP_SIZE)]

best_chrom = None
best_score = -9999.0
best_stats = None
history = []

print("กำลังค้นหาคำตอบที่ดีที่สุดด้วย Genetic Algorithm (Pure GA: ไม่มีรีบาลานซ์)...")

for gen in range(GENS):
    evaluated = [fitness(c) for c in pop]
    scores = [e[0] for e in evaluated]
    max_idx = scores.index(max(scores))
    
    if scores[max_idx] > best_score:
        best_score = scores[max_idx]
        best_chrom = pop[max_idx].copy()
        best_stats = evaluated[max_idx][1]
        
    history.append(best_score)
    
    # การสืบทอดสายพันธุ์ที่ดีที่สุด (Elitism)
    new_pop = [best_chrom.copy()]
    
    while len(new_pop) < POP_SIZE:
        # Selection: Tournament Selection
        i1, i2 = random.sample(range(POP_SIZE), 2)
        p1 = pop[i1] if scores[i1] > scores[i2] else pop[i2]
        i3, i4 = random.sample(range(POP_SIZE), 2)
        p2 = pop[i3] if scores[i3] > scores[i4] else pop[i4]
        
        # Crossover: Single-point Crossover มาตรฐาน (ไม่มีลูปบังคับซ่อมแซมจำนวนคน)
        cut = random.randint(1, N - 1)
        child = p1[:cut] + p2[cut:]
        
        # Mutation: Bit-flip Mutation มาตรฐาน (สุ่มสลับบิตยีน ไม่มีการบังคับ Swap Rebalance)
        for i in range(N):
            if random.random() < 0.015:
                child[i] = 1 - child[i]
                
        new_pop.append(child)
    pop = new_pop

# -------------------------------------------------------------
# 4. แสดงผลการประเมินใน Terminal
# -------------------------------------------------------------
r_skills = [skills[i] for i in range(N) if best_chrom[i] == 0]
g_skills = [skills[i] for i in range(N) if best_chrom[i] == 1]
r_skill_std = float(np.std(r_skills))
g_skill_std = float(np.std(g_skills))
r_female = best_stats['r_count'] - best_stats['r_male']
g_female = best_stats['g_count'] - best_stats['g_male']

print("=" * 68)
print(f" ผลลัพธ์การจัดทีม (Genetic Algorithm - Best Fitness: {best_score:.2f} / 100)")
print("=" * 68)
print(" 1. ความสมดุลขนาดทีม (Team Balance):")
print(f"    - ทีมสีแดง: {best_stats['r_count']} คน | ทีมสีเขียว: {best_stats['g_count']} คน (ผลต่าง: {best_stats['size_diff']} คน)")
print("-" * 68)
print(" 2. ความแฟนเนจด้านกีฬา (Sports Fairness Comparison):")
print(f"    - ทักษะกีฬาเฉลี่ย (Skill): สีแดง {best_stats['r_skill']:.2f} | สีเขียว {best_stats['g_skill']:.2f} (ต่าง {best_stats['diff_skill']:.2f})")
print(f"    - ส่วนเบี่ยงเบนมาตรฐาน  : สีแดง {r_skill_std:.2f} | สีเขียว {g_skill_std:.2f}")
print(f"    - ผู้ชาย (Male)        : สีแดง {best_stats['r_male']} คน | สีเขียว {best_stats['g_male']} คน (ต่าง {best_stats['diff_male']} คน)")
print(f"    - ผู้หญิง (Female)      : สีแดง {r_female} คน | สีเขียว {g_female} คน (ต่าง {abs(r_female - g_female)} คน)")
print(f"    - น้ำหนักเฉลี่ย (Weight) : สีแดง {best_stats['r_w']:.2f} kg | สีเขียว {best_stats['g_w']:.2f} kg (ต่าง {best_stats['diff_w']:.2f} kg)")
print(f"    - ส่วนสูงเฉลี่ย (Height) : สีแดง {best_stats['r_h']:.2f} cm | สีเขียว {best_stats['g_h']:.2f} cm (ต่าง {best_stats['diff_h']:.2f} cm)")
print(f"    - อายุเฉลี่ย (Age)      : สีแดง {best_stats['r_age']:.2f} ปี | สีเขียว {best_stats['g_age']:.2f} ปี (ต่าง {best_stats['diff_age']:.2f} ปี)")
print(f"    - BMI เฉลี่ย            : สีแดง {best_stats['r_bmi']:.2f} | สีเขียว {best_stats['g_bmi']:.2f} (ต่าง {best_stats['diff_bmi']:.2f})")
print("-" * 68)
print(" 3. การกระจายตัวข้ามแผนก (Harmonization across 19 Departments):")
for d in unique_depts:
    cr = sum(1 for i in range(N) if best_chrom[i] == 0 and depts[i] == d)
    cg = sum(1 for i in range(N) if best_chrom[i] == 1 and depts[i] == d)
    print(f"    - แผนก {d:<2} (รวม {cr+cg:<2} คน): สีแดง {cr:<2} คน | สีเขียว {cg:<2} คน")
print("=" * 68)

# -------------------------------------------------------------
# 5. บันทึกผลลัพธ์ลง CSV และไฟล์รายงาน output.txt
# -------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
df['bmi'] = df['bmi'].round(2)
df['Team'] = ['สีแดง (Red)' if v == 0 else 'สีเขียว (Green)' for v in best_chrom]
df.to_csv("outputs/team_assignment.csv", index=False, encoding="utf-8-sig")
print(">> บันทึกรายชื่อผลลัพธ์ลง: outputs/team_assignment.csv")

output_lines = [
    "=" * 75,
    " รายงานผลการจัดแบ่งทีมกีฬา 2 สีด้วย Genetic Algorithm (Pure GA)",
    "=" * 75,
    f"ชุดข้อมูล : data/HR(in).csv (จำนวนสมาชิกทั้งหมด {N} คน, 19 แผนก)",
    f"ผลลัพธ์   : สีแดง {best_stats['r_count']} คน | สีเขียว {best_stats['g_count']} คน (ผลต่าง {best_stats['size_diff']} คน)",
    f"คะแนนความเหมาะสมสูงสุด (Best Fitness): {best_score:.2f} / 100",
    "-" * 75,
    "1. ความแฟนเนจด้านกีฬา (Sports Fairness Comparison):",
    f"   - ทักษะกีฬาเฉลี่ย (Sports Skill) : สีแดง {best_stats['r_skill']:.2f} | สีเขียว {best_stats['g_skill']:.2f} (ต่าง {best_stats['diff_skill']:.2f})",
    f"   - ค่าเบี่ยงเบนมาตรฐาน (Std Dev)  : สีแดง {r_skill_std:.2f} | สีเขียว {g_skill_std:.2f}",
    f"   - สัดส่วนผู้ชาย (Male)           : สีแดง {best_stats['r_male']} คน | สีเขียว {best_stats['g_male']} คน (ต่าง {best_stats['diff_male']} คน)",
    f"   - สัดส่วนผู้หญิง (Female)         : สีแดง {r_female} คน | สีเขียว {g_female} คน (ต่าง {abs(r_female - g_female)} คน)",
    f"   - น้ำหนักเฉลี่ย (Weight)          : สีแดง {best_stats['r_w']:.2f} kg | สีเขียว {best_stats['g_w']:.2f} kg (ต่าง {best_stats['diff_w']:.2f} kg)",
    f"   - ส่วนสูงเฉลี่ย (Height)          : สีแดง {best_stats['r_h']:.2f} cm | สีเขียว {best_stats['g_h']:.2f} cm (ต่าง {best_stats['diff_h']:.2f} cm)",
    f"   - อายุเฉลี่ย (Age)              : สีแดง {best_stats['r_age']:.2f} ปี | สีเขียว {best_stats['g_age']:.2f} ปี (ต่าง {best_stats['diff_age']:.2f} ปี)",
    f"   - ดัชนีมวลกายเฉลี่ย (BMI)         : สีแดง {best_stats['r_bmi']:.2f} | สีเขียว {best_stats['g_bmi']:.2f} (ต่าง {best_stats['diff_bmi']:.2f})",
    "-" * 75,
    "2. การกระจายตัวข้ามแผนก (Harmonization Comparison - ครบทั้ง 19 แผนก):",
]

for d in unique_depts:
    cr = sum(1 for i in range(N) if best_chrom[i] == 0 and depts[i] == d)
    cg = sum(1 for i in range(N) if best_chrom[i] == 1 and depts[i] == d)
    output_lines.append(f"   - แผนก {d:<2} (รวม {cr+cg:<2} คน): สีแดง {cr:<2} คน | สีเขียว {cg:<2} คน")

output_lines.extend([
    "=" * 75,
    f" รายชื่อสมาชิกทีมสีแดง (Team Red: {best_stats['r_count']} คน)",
    "=" * 75,
    f"{'No.':<5} {'Gender':<8} {'Dept':<6} {'Weight':<8} {'Height':<8} {'Age':<6} {'BMI':<6} {'Skill':<6}"
])

red_count = 1
for i in range(N):
    if best_chrom[i] == 0:
        output_lines.append(f"{red_count:<5} {genders[i]:<8} {depts[i]:<6} {weights[i]:<8} {heights[i]:<8} {ages[i]:<6} {bmis[i]:.2f} {skills[i]:.1f}")
        red_count += 1

output_lines.extend([
    "=" * 75,
    f" รายชื่อสมาชิกทีมสีเขียว (Team Green: {best_stats['g_count']} คน)",
    "=" * 75,
    f"{'No.':<5} {'Gender':<8} {'Dept':<6} {'Weight':<8} {'Height':<8} {'Age':<6} {'BMI':<6} {'Skill':<6}"
])

grn_count = 1
for i in range(N):
    if best_chrom[i] == 1:
        output_lines.append(f"{grn_count:<5} {genders[i]:<8} {depts[i]:<6} {weights[i]:<8} {heights[i]:<8} {ages[i]:<6} {bmis[i]:.2f} {skills[i]:.1f}")
        grn_count += 1

full_report_text = "\n".join(output_lines)
with open("outputs/output.txt", "w", encoding="utf-8") as f:
    f.write(full_report_text)
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(full_report_text)
print(">> บันทึกไฟล์สรุปผลรายงาน: outputs/output.txt และ output.txt")

# -------------------------------------------------------------
# 6. สร้างกราฟรวม 3 ด้านในไฟล์เดียว (แม่นยำ ละเอียด ชัดเจน สวยงาม)
# -------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['Leelawadee UI', 'Tahoma', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(18, 11), dpi=300)
gs = gridspec.GridSpec(2, 4, height_ratios=[1.0, 1.15], hspace=0.35, wspace=0.32)

# --- กราฟย่อย 1: Fitness Progression (ตามรูปที่ 2) ---
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(range(len(history)), history, color='blue', linewidth=2.4)
ax1.set_title("Genetic Algorithm: Fitness Progression (HR Data)", fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel("Generation", fontsize=11)
ax1.set_ylabel("Fitness Score", fontsize=11)
ax1.grid(True, linestyle='-', alpha=0.3, color='gray')
ax1.set_xlim(-2, len(history) + 2)

# --- กราฟย่อย 2 (ซ้าย): Sports Skill Distribution - Boxplot + Strip Plot (ตามรูปที่ 3 ด้านซ้าย) ---
ax2 = fig.add_subplot(gs[0, 2])
box_data = [r_skills, g_skills]
bp = ax2.boxplot(box_data, tick_labels=['สีแดง (Red)', 'สีเขียว (Green)'], patch_artist=True, widths=0.52)

colors_box = ['#D9534F', '#4CAE4C']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
    patch.set_edgecolor('black')

for element in ['whiskers', 'caps', 'medians']:
    plt.setp(bp[element], color='black', linewidth=1.2)

# เพิ่ม Scatter Jitter (จุดกระจายตัวของสมาชิกแต่ละคน)
for idx, data_pts in enumerate(box_data, start=1):
    jitter = np.random.normal(0, 0.04, size=len(data_pts))
    ax2.scatter(idx + jitter, data_pts, color='#2F2F2F', alpha=0.35, s=18, edgecolors='none', zorder=3)

ax2.set_title("Sports Skill Distribution (Fairness Check)", fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel("Team", fontsize=10)
ax2.set_ylabel("Sports Skill / Capability", fontsize=10)
ax2.grid(True, axis='y', linestyle='-', alpha=0.3, color='gray')

# --- กราฟย่อย 2 (ขวา): Comparison of Skill Means & Variance (ตามรูปที่ 3 ด้านขวา) ---
ax3 = fig.add_subplot(gs[0, 3])
teams_label = ['สีแดง (Red)', 'สีเขียว (Green)']
means = [best_stats['r_skill'], best_stats['g_skill']]
stds = [r_skill_std, g_skill_std]

x_pos = np.arange(len(teams_label))
bar_width = 0.35

bar_mean = ax3.bar(x_pos - bar_width/2, means, width=bar_width, label='Mean Skill', color='#3498DB', edgecolor='black', alpha=0.9)
bar_std = ax3.bar(x_pos + bar_width/2, stds, width=bar_width, label='Std Dev (Spread)', color='#F39C12', edgecolor='black', alpha=0.9)

ax3.set_xticks(x_pos)
ax3.set_xticklabels(teams_label, fontsize=9.5)
ax3.set_title("Comparison of Skill Means & Variance", fontsize=11, fontweight='bold', pad=10)
ax3.set_ylabel("Score", fontsize=10)
ax3.set_ylim(0, max(means) * 1.25)
ax3.legend(loc='upper right', fontsize=8.5)
ax3.grid(True, axis='y', linestyle='-', alpha=0.3, color='gray')

# ใส่ตัวเลขระบุบนแท่งกราฟ Mean และ Std
for b in bar_mean:
    val = b.get_height()
    ax3.annotate(f"{val:.1f}", xy=(b.get_x() + b.get_width() / 2, val),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for b in bar_std:
    val = b.get_height()
    ax3.annotate(f"{val:.1f}", xy=(b.get_x() + b.get_width() / 2, val),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# --- กราฟย่อย 3: Cross-Department Harmonization across Teams (ตามรูปที่ 1) ---
ax4 = fig.add_subplot(gs[1, :])
dept_r = [sum(1 for i in range(N) if best_chrom[i] == 0 and depts[i] == d) for d in unique_depts]
dept_g = [sum(1 for i in range(N) if best_chrom[i] == 1 and depts[i] == d) for d in unique_depts]

x_dept = np.arange(len(unique_depts))
bar_w = 0.38

b_dept_r = ax4.bar(x_dept - bar_w/2, dept_r, width=bar_w, label='สีแดง (Red)', color='#E55353', edgecolor='black', alpha=0.88)
b_dept_g = ax4.bar(x_dept + bar_w/2, dept_g, width=bar_w, label='สีเขียว (Green)', color='#5CB85C', edgecolor='black', alpha=0.88)

ax4.set_xticks(x_dept)
ax4.set_xticklabels([f"Dept {d}\n(รวม {dept_totals[d]})" for d in unique_depts], fontsize=9)
ax4.set_ylim(0, max(dept_r + dept_g) + 3)
ax4.set_title("Cross-Department Harmonization: Distribution across Teams", fontsize=13, fontweight='bold', pad=12)
ax4.set_xlabel("Department", fontsize=11, labelpad=8)
ax4.set_ylabel("Number of Members", fontsize=11)
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, axis='y', linestyle='-', alpha=0.3, color='gray')

# ใส่ตัวเลขจำนวนคนบนแต่ละแท่งของทุกแผนก
for b in b_dept_r:
    val = b.get_height()
    ax4.annotate(f"{int(val)}", xy=(b.get_x() + b.get_width() / 2, val),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
for b in b_dept_g:
    val = b.get_height()
    ax4.annotate(f"{int(val)}", xy=(b.get_x() + b.get_width() / 2, val),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plot_path = os.path.abspath("outputs/summary_plots.png")
fig.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f">> บันทึกกราฟความละเอียดสูง (300 DPI) สำเร็จที่: {plot_path}")

# เปิดแสดงผลรูปภาพกราฟขึ้นมาบนหน้าจอ
try:
    os.startfile(plot_path)
except Exception:
    pass

plt.close()
