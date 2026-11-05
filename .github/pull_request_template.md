## 📌 คำอธิบายการเปลี่ยนแปลง (Summary of Changes)
- ออกแบบและพัฒนาอัลกอริทึม **Genetic Algorithm (GA)** สำหรับแบ่งสมาชิกออกเป็น 2 สี (สีแดง / สีเขียว)
- กำหนด **Chromosome Representation**: Binary array ขนาด $N$ ($0=$ สีแดง, $1=$ สีเขียว) พร้อมฟังก์ชันคงความสมดุลขนาดทีม
- ออกแบบ **Multi-Objective Fitness Function**:
  1. ความเท่าเทียม/แฟนเนจทางกีฬา (**Sports Fairness & Engagement Score**)
  2. การสร้างความรู้จักและการกระจายตัวข้ามแผนก (**Cross-department Harmonization Score**)
- เพิ่มชุดทดสอบ Unit Tests (`pytest`) และโมดูลสร้าง Visualization กราฟสรุปผล (Convergence, Dept Distribution, Sports Balance)

---

## 🔗 Issues ที่เกี่ยวข้อง (Related Issues)
Closes #1

---

## 🧪 การทดสอบ (Testing & Verification)
- [x] รัน Unit Tests ผ่านทั้งหมด 6/6 รายการ (`pytest tests/test_ga.py`)
- [x] ตรวจสอบความสมดุลของจำนวนสมาชิกทั้ง 2 สี (เท่ากัน 20:20 หรือ ต่างกันไม่เกิน 1 คน)
- [x] ตรวจสอบกราฟสรุปผลในโฟลเดอร์ `outputs/`

---

## 📊 ผลการประเมิน (Evaluation Metrics)
- **Total Fitness**: ~99.1 / 100
- **Sports Fairness Score**: ~99.7 / 100 (ผลต่างค่าเฉลี่ยสกิลกีฬาเพียง 0.05 แต้ม)
- **Harmonize Score**: ~98.4 / 100 (กระจายสมาชิกทุกแผนกลงทั้ง 2 สีอย่างทั่วถึง)
