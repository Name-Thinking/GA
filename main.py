import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# กำหนดค่า Seed เพื่อให้ผลการทดลองคงที่ แม่นยำ และตรวจสอบซ้ำได้ (Reproducibility)
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. โหลดและเตรียมชุดข้อมูลพนักงานระดับองค์กรจาก data/HR(in).csv (1,000 คน)
# ---------------------------------------------------------------------------
df = pd.read_csv("data/HR(in).csv")
df['department'] = df['department'].astype(str).str.strip()

if 'bmi' not in df.columns:
    df['bmi'] = (df['weight'] / ((df['height'] / 100.0) ** 2)).round(2)

if 'sports_skill' not in df.columns:
    bmi_score = 100 - np.clip(np.abs(df['bmi'] - 22.0) * 4.0, 0, 50)
    age_score = 100 - np.clip((df['age'] - 20) * 1.5, 0, 50)
    height_score = np.clip((df['height'] - 140) / (190 - 140) * 100, 30, 100)
    df['sports_skill'] = (0.45 * bmi_score + 0.35 * age_score + 0.20 * height_score).round(1)

N = len(df)
skills = df['sports_skill'].to_numpy(dtype=float)
ages = df['age'].to_numpy(dtype=float)
weights = df['weight'].to_numpy(dtype=float)
heights = df['height'].to_numpy(dtype=float)
bmis = df['bmi'].to_numpy(dtype=float)
genders = df['gender'].tolist()
is_male = (df['gender'] == 'Male').to_numpy(dtype=int)
depts = df['department'].tolist()
emp_ids = df['emp_id'].tolist() if 'emp_id' in df.columns else [f"EMP{i+1:04d}" for i in range(N)]

unique_depts = sorted(set(depts))
dept_masks = [np.array([1 if d == ud else 0 for d in depts], dtype=int) for ud in unique_depts]
dept_totals = {ud: int(mask.sum()) for ud, mask in zip(unique_depts, dept_masks)}

# ---------------------------------------------------------------------------
# 2. ฟังก์ชันประเมินความเหมาะสมและความแม่นยำ (Fitness & Multi-Objective Function)
#    * Pure Genetic Algorithm: ไม่มีการใช้ลูปซ่อมแซมหรือบังคับรีบาลานซ์
#    * คำนวณจากความคลาดเคลื่อนจริงทั้งด้านกีฬา/สรีระ และการกระจายคนข้ามแผนก
# ---------------------------------------------------------------------------
def fitness(chrom_arr):
    g_mask = chrom_arr == 1
    r_mask = ~g_mask
    r_count = int(r_mask.sum())
    g_count = N - r_count

    if r_count == 0 or g_count == 0:
        return -9999.0, None

    # 1) ความสมดุลของจำนวนสมาชิกในแต่ละทีม (Team Size Balance)
    size_diff = abs(r_count - g_count)

    # 2) ความแฟนเนจด้านกีฬาและสมรรถภาพทางกาย (Sports Fairness & Biometrics)
    r_s = skills[r_mask]
    g_s = skills[g_mask]
    r_skill = float(r_s.mean())
    g_skill = float(g_s.mean())
    diff_skill = abs(r_skill - g_skill)

    r_std = float(r_s.std())
    g_std = float(g_s.std())
    diff_std = abs(r_std - g_std)

    r_male = int(is_male[r_mask].sum())
    g_male = int(is_male[g_mask].sum())
    diff_male = abs(r_male - g_male)

    r_age = float(ages[r_mask].mean())
    g_age = float(ages[g_mask].mean())
    diff_age = abs(r_age - g_age)

    r_w = float(weights[r_mask].mean())
    g_w = float(weights[g_mask].mean())
    diff_w = abs(r_w - g_w)

    r_h = float(heights[r_mask].mean())
    g_h = float(heights[g_mask].mean())
    diff_h = abs(r_h - g_h)

    r_bmi = float(bmis[r_mask].mean())
    g_bmi = float(bmis[g_mask].mean())
    diff_bmi = abs(r_bmi - g_bmi)

    # 3) การกระจายสมาชิกข้ามแผนก (Cross-Department Harmonization)
    raw_dept_diff = 0
    for ud, d_mask in zip(unique_depts, dept_masks):
        cr = int(d_mask[r_mask].sum())
        cg = dept_totals[ud] - cr
        raw_dept_diff += abs(cr - cg)

    # คะแนนความเหมาะสมรวม (เต็ม 100 คะแนน)
    score = (100.0
             - (size_diff * 1.2)
             - (diff_skill * 2.5)
             - (diff_std * 2.0)
             - (diff_male * 1.0)
             - (diff_age * 1.5)
             - (diff_w * 0.8)
             - (diff_h * 0.8)
             - (diff_bmi * 2.0)
             - (raw_dept_diff * 1.2))

    stats = {
        'r_count': r_count, 'g_count': g_count, 'size_diff': size_diff,
        'r_skill': r_skill, 'g_skill': g_skill, 'diff_skill': diff_skill,
        'r_std': r_std, 'g_std': g_std, 'diff_std': diff_std,
        'r_male': r_male, 'g_male': g_male, 'diff_male': diff_male,
        'r_age': r_age, 'g_age': g_age, 'diff_age': diff_age,
        'r_w': r_w, 'g_w': g_w, 'diff_w': diff_w,
        'r_h': r_h, 'g_h': g_h, 'diff_h': diff_h,
        'r_bmi': r_bmi, 'g_bmi': g_bmi, 'diff_bmi': diff_bmi,
        'dept_diff': raw_dept_diff
    }
    return score, stats

# ---------------------------------------------------------------------------
# 3. กระบวนการเรียนรู้ของ Genetic Algorithm (Pure GA Optimization)
# ---------------------------------------------------------------------------
POP_SIZE, GENS = 100, 120

pop = [np.random.randint(0, 2, size=N) for _ in range(POP_SIZE)]
best_chrom = None
best_score = -9999.0
best_stats = None
history = []

print(f"กำลังประมวลผล Genetic Algorithm บนชุดข้อมูลพนักงาน {N:,} คน ({len(unique_depts)} แผนก)...")

for gen in range(GENS):
    evaluated = [fitness(c) for c in pop]
    scores = [e[0] for e in evaluated]
    max_idx = int(np.argmax(scores))

    if scores[max_idx] > best_score:
        best_score = scores[max_idx]
        best_chrom = pop[max_idx].copy()
        best_stats = evaluated[max_idx][1]

    history.append(best_score)

    # Elitism: รักษาโครโมโซมที่ดีที่สุด 4 อันดับแรกไว้ในรุ่นถัดไป
    sorted_indices = np.argsort(scores)[::-1]
    new_pop = [pop[i].copy() for i in sorted_indices[:4]]

    while len(new_pop) < POP_SIZE:
        # Selection: Tournament Selection (k=3)
        c1 = random.sample(range(POP_SIZE), 3)
        p1 = pop[max(c1, key=lambda k: scores[k])]
        c2 = random.sample(range(POP_SIZE), 3)
        p2 = pop[max(c2, key=lambda k: scores[k])]

        # Crossover: Two-point Crossover มาตรฐาน
        pt1, pt2 = sorted(random.sample(range(1, N), 2))
        child = np.concatenate([p1[:pt1], p2[pt1:pt2], p1[pt2:]])

        # Mutation: Bit-flip Mutation มาตรฐาน (อัตรา ~1/N)
        if random.random() < 0.85:
            idx = random.randint(0, N - 1)
            child[idx] = 1 - child[idx]

        new_pop.append(child)
    pop = new_pop

# ---------------------------------------------------------------------------
# 4. คำนวณดัชนีชี้วัดคุณภาพและความแม่นยำตามหลักวิชาการ (Evaluation Metrics)
# ---------------------------------------------------------------------------
r_skills = skills[best_chrom == 0]
g_skills = skills[best_chrom == 1]
r_female = best_stats['r_count'] - best_stats['r_male']
g_female = best_stats['g_count'] - best_stats['g_male']

sports_penalty = (best_stats['diff_skill'] * 2.5 + best_stats['diff_std'] * 2.0 +
                  best_stats['diff_male'] * 1.0 + best_stats['diff_age'] * 1.5 +
                  best_stats['diff_w'] * 0.8 + best_stats['diff_h'] * 0.8 +
                  best_stats['diff_bmi'] * 2.0)
sports_accuracy = max(0.0, 100.0 - sports_penalty * 2.5)
dept_accuracy = max(0.0, 100.0 - (best_stats['dept_diff'] * 1.8))

even_dept_perfect = sum(
    1 for ud, d_mask in zip(unique_depts, dept_masks)
    if dept_totals[ud] % 2 == 0 and int(d_mask[best_chrom == 0].sum()) * 2 == dept_totals[ud]
)
total_even_depts = sum(1 for ud in unique_depts if dept_totals[ud] % 2 == 0)

print("=" * 75)
print(f" รายงานผลการจัดแบ่งทีมกีฬาด้วย Genetic Algorithm (จำนวน {N:,} คน, {len(unique_depts)} แผนก)")
print(f" คะแนนความเหมาะสมและความแม่นยำรวม (Best Fitness Score): {best_score:.2f} / 100 ({best_score:.2f}%)")
print("=" * 75)
print(" 1. ดัชนีวัดผลความสมดุลและความแม่นยำ (Performance & Evaluation Metrics):")
print(f"    - คะแนนความแม่นยำรวม (Overall Fitness Score)   : {best_score:.2f}%")
print(f"    - ความสมดุลด้านสมรรถภาพกีฬา (Sports Fairness)  : {sports_accuracy:.2f}%")
print(f"    - ความสมดุลด้านการกระจายแผนก (Harmonization)   : {dept_accuracy:.2f}%")
print(f"    - แผนกจำนวนคนคู่ที่แบ่งครึ่ง 50:50 ลงตัวสมบูรณ์    : {even_dept_perfect} / {total_even_depts} แผนก")
print("-" * 75)
print(" 2. เปรียบเทียบความแฟนเนจด้านกีฬาและสรีระ (Sports Fairness Comparison):")
print(f"    - จำนวนสมาชิกในทีม          : สีแดง {best_stats['r_count']} คน | สีเขียว {best_stats['g_count']} คน (ผลต่าง {best_stats['size_diff']} คน)")
print(f"    - ทักษะกีฬาเฉลี่ย (Mean)    : สีแดง {best_stats['r_skill']:.2f} | สีเขียว {best_stats['g_skill']:.2f} (ผลต่าง {best_stats['diff_skill']:.2f})")
print(f"    - ส่วนเบี่ยงเบนมาตรฐาน (SD) : สีแดง {best_stats['r_std']:.2f} | สีเขียว {best_stats['g_std']:.2f} (ผลต่าง {best_stats['diff_std']:.2f})")
print(f"    - สัดส่วนพนักงานชาย (Male)   : สีแดง {best_stats['r_male']} คน | สีเขียว {best_stats['g_male']} คน (ผลต่าง {best_stats['diff_male']} คน)")
print(f"    - สัดส่วนพนักงานหญิง (Female): สีแดง {r_female} คน | สีเขียว {g_female} คน (ผลต่าง {abs(r_female - g_female)} คน)")
print(f"    - น้ำหนักเฉลี่ย (Weight)     : สีแดง {best_stats['r_w']:.2f} kg | สีเขียว {best_stats['g_w']:.2f} kg (ผลต่าง {best_stats['diff_w']:.2f} kg)")
print(f"    - ส่วนสูงเฉลี่ย (Height)     : สีแดง {best_stats['r_h']:.2f} cm | สีเขียว {best_stats['g_h']:.2f} cm (ผลต่าง {best_stats['diff_h']:.2f} cm)")
print(f"    - อายุเฉลี่ย (Age)          : สีแดง {best_stats['r_age']:.2f} ปี | สีเขียว {best_stats['g_age']:.2f} ปี (ผลต่าง {best_stats['diff_age']:.2f} ปี)")
print(f"    - ดัชนีมวลกายเฉลี่ย (BMI)    : สีแดง {best_stats['r_bmi']:.2f} | สีเขียว {best_stats['g_bmi']:.2f} (ผลต่าง {best_stats['diff_bmi']:.2f})")
print("-" * 75)
print(f" 3. การกระจายสมาชิกข้ามสายงาน (Cross-Department Harmonization - {len(unique_depts)} แผนก):")
for ud, d_mask in zip(unique_depts, dept_masks):
    cr = int(d_mask[best_chrom == 0].sum())
    cg = dept_totals[ud] - cr
    status = "สมดุล 50:50" if cr == cg else f"ผลต่าง {abs(cr - cg)} คน (แผนกเลขคี่)"
    print(f"    - {ud:<18} (รวม {cr+cg:<3} คน): สีแดง {cr:<3} คน | สีเขียว {cg:<3} คน [{status}]")
print("=" * 75)

# ---------------------------------------------------------------------------
# 5. บันทึกไฟล์ผลลัพธ์ CSV และรายงานสรุปฉบับสมบูรณ์ (output.txt)
# ---------------------------------------------------------------------------
os.makedirs("outputs", exist_ok=True)
out_df = df.copy()
out_df['Team'] = ['สีแดง (Red)' if v == 0 else 'สีเขียว (Green)' for v in best_chrom]
out_df.to_csv("outputs/team_assignment.csv", index=False, encoding="utf-8-sig")
print(">> บันทึกตารางจัดทีมพนักงานลงไฟล์: outputs/team_assignment.csv")

output_lines = [
    "=" * 80,
    " รายงานผลการจัดแบ่งทีมกีฬาองค์กร 2 สีด้วย Genetic Algorithm (Enterprise HR Edition)",
    "=" * 80,
    f"ชุดข้อมูลต้นฉบับ : data/HR(in).csv (จำนวนพนักงานทั้งหมด {N:,} คน, {len(unique_depts)} แผนกสายงาน)",
    f"ผลการแบ่งทีม    : ทีมสีแดง {best_stats['r_count']} คน | ทีมสีเขียว {best_stats['g_count']} คน (ผลต่างขนาดทีม {best_stats['size_diff']} คน)",
    f"คะแนนความเหมาะสมและความแม่นยำรวม (Best Fitness Score): {best_score:.2f} / 100 ({best_score:.2f}%)",
    "-" * 80,
    "1. ดัชนีวัดผลความสมดุลและความแม่นยำ (Performance & Evaluation Metrics):",
    f"   - คะแนนความเหมาะสมรวม (Overall Fitness Score)   : {best_score:.2f}% (ประเมินจากผลต่างจริงทุกมิติ)",
    f"   - ความสมดุลด้านสมรรถภาพกีฬา (Sports Fairness)  : {sports_accuracy:.2f}%",
    f"   - ความสมดุลด้านการกระจายแผนก (Harmonization)   : {dept_accuracy:.2f}%",
    f"   - แผนกจำนวนคนคู่ที่แบ่งครึ่ง 50:50 ลงตัวสมบูรณ์    : {even_dept_perfect} / {total_even_depts} แผนก",
    "-" * 80,
    "2. เปรียบเทียบความแฟนเนจด้านกีฬาและสรีระ (Sports Fairness Comparison):",
    f"   - ทักษะกีฬาเฉลี่ย (Sports Skill) : สีแดง {best_stats['r_skill']:.2f} | สีเขียว {best_stats['g_skill']:.2f} (ต่าง {best_stats['diff_skill']:.2f})",
    f"   - ส่วนเบี่ยงเบนมาตรฐาน (Std Dev) : สีแดง {best_stats['r_std']:.2f} | สีเขียว {best_stats['g_std']:.2f} (ต่าง {best_stats['diff_std']:.2f})",
    f"   - สัดส่วนพนักงานชาย (Male)       : สีแดง {best_stats['r_male']} คน | สีเขียว {best_stats['g_male']} คน (ต่าง {best_stats['diff_male']} คน)",
    f"   - สัดส่วนพนักงานหญิง (Female)     : สีแดง {r_female} คน | สีเขียว {g_female} คน (ต่าง {abs(r_female - g_female)} คน)",
    f"   - น้ำหนักเฉลี่ย (Weight)         : สีแดง {best_stats['r_w']:.2f} kg | สีเขียว {best_stats['g_w']:.2f} kg (ต่าง {best_stats['diff_w']:.2f} kg)",
    f"   - ส่วนสูงเฉลี่ย (Height)         : สีแดง {best_stats['r_h']:.2f} cm | สีเขียว {best_stats['g_h']:.2f} cm (ต่าง {best_stats['diff_h']:.2f} cm)",
    f"   - อายุเฉลี่ย (Age)              : สีแดง {best_stats['r_age']:.2f} ปี | สีเขียว {best_stats['g_age']:.2f} ปี (ต่าง {best_stats['diff_age']:.2f} ปี)",
    f"   - ดัชนีมวลกายเฉลี่ย (BMI)        : สีแดง {best_stats['r_bmi']:.2f} | สีเขียว {best_stats['g_bmi']:.2f} (ต่าง {best_stats['diff_bmi']:.2f})",
    "-" * 80,
    f"3. การกระจายตัวข้ามสายงาน (Cross-Department Harmonization - ครบทั้ง {len(unique_depts)} แผนก):",
]

for ud, d_mask in zip(unique_depts, dept_masks):
    cr = int(d_mask[best_chrom == 0].sum())
    cg = dept_totals[ud] - cr
    status = "สมดุล 50:50" if cr == cg else f"ต่าง {abs(cr - cg)} คน (เลขคี่)"
    output_lines.append(f"   - {ud:<18} (รวม {cr+cg:<3} คน): สีแดง {cr:<3} คน | สีเขียว {cg:<3} คน [{status}]")

output_lines.extend([
    "=" * 80,
    f" รายชื่อสมาชิกทีมสีแดง (Team Red: {best_stats['r_count']} คน)",
    "=" * 80,
    f"{'No.':<6} {'EmpID':<9} {'Gender':<8} {'Department':<20} {'Weight':<8} {'Height':<8} {'Age':<6} {'BMI':<7} {'Skill':<6}"
])

red_no = 1
for i in range(N):
    if best_chrom[i] == 0:
        output_lines.append(
            f"{red_no:<6} {emp_ids[i]:<9} {genders[i]:<8} {depts[i]:<20} "
            f"{int(weights[i]):<8} {int(heights[i]):<8} {int(ages[i]):<6} {bmis[i]:<7.2f} {skills[i]:<6.1f}"
        )
        red_no += 1

output_lines.extend([
    "=" * 80,
    f" รายชื่อสมาชิกทีมสีเขียว (Team Green: {best_stats['g_count']} คน)",
    "=" * 80,
    f"{'No.':<6} {'EmpID':<9} {'Gender':<8} {'Department':<20} {'Weight':<8} {'Height':<8} {'Age':<6} {'BMI':<7} {'Skill':<6}"
])

grn_no = 1
for i in range(N):
    if best_chrom[i] == 1:
        output_lines.append(
            f"{grn_no:<6} {emp_ids[i]:<9} {genders[i]:<8} {depts[i]:<20} "
            f"{int(weights[i]):<8} {int(heights[i]):<8} {int(ages[i]):<6} {bmis[i]:<7.2f} {skills[i]:<6.1f}"
        )
        grn_no += 1

full_report_text = "\n".join(output_lines)
with open("outputs/output.txt", "w", encoding="utf-8") as f:
    f.write(full_report_text)
with open("output.txt", "w", encoding="utf-8") as f:
    f.write(full_report_text)
print(">> บันทึกรายงานสรุปฉบับสมบูรณ์: outputs/output.txt และ output.txt")

# ---------------------------------------------------------------------------
# 6. สร้างกราฟ Dashboard รวม 3 มิติในไฟล์เดียว (ความละเอียดสูง 300 DPI)
# ---------------------------------------------------------------------------
plt.rcParams['font.sans-serif'] = ['Leelawadee UI', 'Tahoma', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig = plt.figure(figsize=(18, 11), dpi=300)
gs = gridspec.GridSpec(2, 4, height_ratios=[1.0, 1.18], hspace=0.36, wspace=0.32)

# --- กราฟที่ 1: Genetic Algorithm Fitness Progression ---
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(range(len(history)), history, color='#0000FF', linewidth=2.5)
ax1.set_title("Genetic Algorithm: Fitness Progression (HR Data)", fontsize=13, fontweight='bold', pad=10)
ax1.set_xlabel("Generation", fontsize=11)
ax1.set_ylabel("Fitness Score", fontsize=11)
ax1.grid(True, linestyle='-', alpha=0.3, color='gray')
ax1.set_xlim(-2, len(history) + 2)

# --- กราฟที่ 2 (ซ้าย): Sports Skill Distribution (Fairness Check) ---
ax2 = fig.add_subplot(gs[0, 2])
box_data = [r_skills, g_skills]
bp = ax2.boxplot(box_data, tick_labels=['สีแดง (Red)', 'สีเขียว (Green)'], patch_artist=True, widths=0.52)

for patch, color in zip(bp['boxes'], ['#D9534F', '#4CAE4C']):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
    patch.set_edgecolor('black')

for element in ['whiskers', 'caps', 'medians']:
    plt.setp(bp[element], color='black', linewidth=1.2)

for idx, data_pts in enumerate(box_data, start=1):
    jitter = np.random.normal(0, 0.045, size=len(data_pts))
    ax2.scatter(idx + jitter, data_pts, color='#2B2B2B', alpha=0.22, s=14, edgecolors='none', zorder=3)

ax2.set_title("Sports Skill Distribution (Fairness Check)", fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel("Team", fontsize=10)
ax2.set_ylabel("Sports Skill / Capability", fontsize=10)
ax2.grid(True, axis='y', linestyle='-', alpha=0.3, color='gray')

# --- กราฟที่ 2 (ขวา): Comparison of Skill Means & Variance ---
ax3 = fig.add_subplot(gs[0, 3])
teams_label = ['สีเขียว (Green)', 'สีแดง (Red)']
means = [best_stats['g_skill'], best_stats['r_skill']]
stds = [best_stats['g_std'], best_stats['r_std']]

x_pos = np.arange(len(teams_label))
bar_width = 0.35

bar_mean = ax3.bar(x_pos - bar_width/2, means, width=bar_width, label='Mean Skill', color='#3498DB', edgecolor='black', alpha=0.92)
bar_std = ax3.bar(x_pos + bar_width/2, stds, width=bar_width, label='Std Dev (Spread)', color='#F39C12', edgecolor='black', alpha=0.92)

ax3.set_xticks(x_pos)
ax3.set_xticklabels(teams_label, fontsize=9.5)
ax3.set_title("Comparison of Skill Means & Variance", fontsize=11, fontweight='bold', pad=10)
ax3.set_ylabel("Score", fontsize=10)
ax3.set_ylim(0, max(means) * 1.25)
ax3.legend(loc='upper right', fontsize=8.5)
ax3.grid(True, axis='y', linestyle='-', alpha=0.3, color='gray')

for b in bar_mean:
    val = b.get_height()
    ax3.annotate(f"{val:.1f}", xy=(b.get_x() + b.get_width() / 2, val),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')
for b in bar_std:
    val = b.get_height()
    ax3.annotate(f"{val:.1f}", xy=(b.get_x() + b.get_width() / 2, val),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

# --- กราฟที่ 3: Cross-Department Harmonization across Teams ---
ax4 = fig.add_subplot(gs[1, :])
dept_r = [int(d_mask[best_chrom == 0].sum()) for d_mask in dept_masks]
dept_g = [dept_totals[ud] - cr for ud, cr in zip(unique_depts, dept_r)]

x_dept = np.arange(len(unique_depts))
bar_w = 0.36

b_dept_r = ax4.bar(x_dept - bar_w/2, dept_r, width=bar_w, label='สีแดง (Red)', color='#E55353', edgecolor='black', alpha=0.9)
b_dept_g = ax4.bar(x_dept + bar_w/2, dept_g, width=bar_w, label='สีเขียว (Green)', color='#5CB85C', edgecolor='black', alpha=0.9)

ax4.set_xticks(x_dept)
ax4.set_xticklabels(unique_depts, fontsize=10, rotation=15, ha='right')
ax4.set_ylim(0, max(dept_r + dept_g) * 1.18)
ax4.set_title("Cross-Department Harmonization: Distribution across Teams", fontsize=13, fontweight='bold', pad=12)
ax4.set_xlabel("Department", fontsize=11, labelpad=6)
ax4.set_ylabel("Number of Members", fontsize=11)
ax4.legend(loc='upper right', fontsize=10)
ax4.grid(True, axis='y', linestyle='-', alpha=0.3, color='gray')

for b in b_dept_r:
    val = b.get_height()
    ax4.annotate(f"{int(val)}", xy=(b.get_x() + b.get_width() / 2, val),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')
for b in b_dept_g:
    val = b.get_height()
    ax4.annotate(f"{int(val)}", xy=(b.get_x() + b.get_width() / 2, val),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plot_path = os.path.abspath("outputs/summary_plots.png")
if os.path.exists(plot_path):
    try:
        os.remove(plot_path)
    except Exception:
        pass
fig.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f">> บันทึกกราฟความละเอียดสูง (300 DPI) สำเร็จที่: {plot_path}")

try:
    os.startfile(plot_path)
except Exception:
    pass

plt.close()
