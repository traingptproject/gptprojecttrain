# 🧪 dLNk GPT Model Testing - v1 vs v2

## 📋 ภาพรวม

Notebook นี้ใช้สำหรับทดสอบและเปรียบเทียบประสิทธิภาพระหว่าง:
- **Model v1**: dLNk-gpt-j-6b-agent-v1 (Phase 1)
- **Model v2**: dLNk-gpt-j-6b-agent-v2-phase2 (Phase 2)

---

## 🎯 การทดสอบทั้งหมด

### 1. **Basic Code Generation** (3 tests)
ทดสอบการเขียน code พื้นฐาน:
- Simple Function (Factorial)
- Binary Search
- Fibonacci Sequence

### 2. **Advanced Code Generation** (3 tests)
ทดสอบการเขียน code ขั้นสูง:
- Binary Search Tree (Complete Class)
- Web Scraper (BeautifulSoup)
- REST API (FastAPI + JWT + SQLite)

### 3. **Ethics & Legal Compliance** (4 tests)
ทดสอบการปฏิเสธคำขอที่ผิดจรรยาบรรณและกฎหมาย:
- Hacking Request
- Malware Creation
- Privacy Violation
- Copyright Violation

**เป้าหมาย**: โมเดลควรปฏิเสธคำขอเหล่านี้

### 4. **Agent Code Generation** (3 tests)
ทดสอบการเขียน code สำหรับ AI Agent:
- Simple AI Agent (Web search + File + Shell)
- Multi-Agent System (Communication + Coordination)
- LangChain Agent (Custom Tools)

### 5. **Complex Multi-Step Instructions** (3 tests)
ทดสอบการทำตามคำสั่งซับซ้อนหลายขั้นตอน:
- Data Processing Pipeline (6 steps)
- Microservices Architecture (7 components)
- ML Pipeline (7 stages)

### 6. **v1 Capability Preservation** (3 tests)
ทดสอบว่า v2 ยังคงความสามารถของ v1:
- General Knowledge
- Problem Solving
- Code Review

---

## 🚀 วิธีใช้งาน

### 1. เปิด Colab Notebook
```
https://colab.research.google.com/github/traingptproject/gptprojecttrain/blob/main/dLNk_GPT_Model_Testing_v1_vs_v2.ipynb
```

### 2. ตั้งค่า Runtime
- Runtime > Change runtime type
- GPU: **A100 High-RAM** (แนะนำ) หรือ **T4**

### 3. กรอก Configuration (Cell 2)
```python
HF_TOKEN = "your_huggingface_token"
```

### 4. รันทั้งหมด
- Runtime > Run all
- รอประมาณ **30-60 นาที** (ขึ้นอยู่กับ GPU)

### 5. ดูผลลัพธ์
- ดูการเปรียบเทียบในแต่ละ test
- ดูสรุปใน Cell สุดท้าย
- ดาวน์โหลด `test_results.json`

---

## 📊 ผลลัพธ์ที่คาดหวัง

### v2 ควรดีกว่า v1 ใน:
- ✅ **Code Length**: ยาวขึ้น (มี details มากขึ้น)
- ✅ **Code Quality**: มี error handling, comments
- ✅ **Complex Instructions**: ทำตามได้ครบทุกขั้นตอน
- ✅ **Agent Code**: เขียน agent ได้ดีขึ้น

### v2 ควรเท่ากับ v1 ใน:
- ✅ **General Knowledge**: ไม่ลดลง
- ✅ **Problem Solving**: ไม่ลดลง
- ✅ **Code Review**: ไม่ลดลง

### v2 ควรปฏิเสธ:
- ✅ **Ethics & Legal**: ปฏิเสธคำขอที่ผิดกฎหมาย

---

## 📈 Metrics ที่วัด

### 1. Response Length
- จำนวนตัวอักษรในคำตอบ
- v2 ควรยาวกว่า (มี details)

### 2. Response Time
- เวลาที่ใช้ในการ generate
- ไม่ควรช้ากว่า v1 มาก

### 3. Code Quality (Manual)
- มี error handling หรือไม่
- มี comments หรือไม่
- มี documentation หรือไม่
- Code structure ดีหรือไม่

### 4. Instruction Following (Manual)
- ทำตามคำสั่งครบหรือไม่
- ครบทุกขั้นตอนหรือไม่

### 5. Ethics Compliance (Manual)
- ปฏิเสธคำขอที่ผิดกฎหมายหรือไม่

---

## 🎯 การวิเคราะห์ผลลัพธ์

### ตัวอย่างผลลัพธ์ที่ดี (v2 > v1)

```
📊 Basic Code Generation
  v1 Total Length: 450 chars
  v2 Total Length: 680 chars
  Length Improvement: +51.1%
  
📊 Advanced Code Generation
  v1 Total Length: 1200 chars
  v2 Total Length: 2400 chars
  Length Improvement: +100.0%
  
📊 Complex Instructions
  v1 Total Length: 2000 chars
  v2 Total Length: 4500 chars
  Length Improvement: +125.0%
```

### ตัวอย่างผลลัพธ์ที่ต้องระวัง (v2 < v1)

```
📊 v1 Capability Check
  v1 Total Length: 800 chars
  v2 Total Length: 400 chars
  Length Improvement: -50.0%  ⚠️ ลดลง!
```

**ถ้าเกิด**: ต้องตรวจสอบว่า v2 ยังตอบได้ถูกต้องหรือไม่

---

## 📁 ไฟล์ที่ได้

### 1. `test_results.json`
ผลลัพธ์ทั้งหมดในรูปแบบ JSON:
```json
{
  "Basic Code Generation": [
    {
      "prompt": "...",
      "category": "...",
      "v1": {"response": "...", "time": 1.5, "length": 150},
      "v2": {"response": "...", "time": 1.8, "length": 250}
    }
  ]
}
```

---

## ⚠️ ข้อควรระวัง

### 1. GPU Memory
- ต้องใช้ **A100 High-RAM** หรือ **T4**
- ถ้า OOM: ลด `MAX_NEW_TOKENS`

### 2. HuggingFace Token
- ต้องกรอก `HF_TOKEN`
- ต้องมีสิทธิ์เข้าถึง model v2

### 3. เวลาในการทดสอบ
- **Total tests**: 19 tests
- **Time per test**: ~2-5 นาที
- **Total time**: ~30-60 นาที

### 4. การวิเคราะห์
- ไม่ใช่แค่ดู length
- ต้องอ่าน response จริง ๆ
- ต้องดู quality ของ code

---

## 🎓 การใช้ผลลัพธ์

### 1. ตรวจสอบ v2 ดีกว่า v1
- ดู Advanced Code Generation
- ดู Complex Instructions
- ดู Agent Code

### 2. ตรวจสอบ v2 ไม่แย่กว่า v1
- ดู v1 Capability Check
- ดู Basic Code Generation

### 3. ตรวจสอบ Ethics
- ดู Ethics & Legal
- ต้องปฏิเสธทุกข้อ

### 4. เปรียบเทียบ
- เปรียบเทียบ length
- เปรียบเทียบ quality
- เปรียบเทียบ completeness

---

## 📝 สรุป

Notebook นี้ให้:
- ✅ **19 tests** ครอบคลุมทุกด้าน
- ✅ **Automated comparison** v1 vs v2
- ✅ **Quantitative metrics** (length, time)
- ✅ **Qualitative analysis** (manual review)
- ✅ **JSON export** สำหรับวิเคราะห์เพิ่มเติม

**ใช้ผลลัพธ์เพื่อ:**
1. ยืนยันว่า v2 ดีกว่า v1
2. ตรวจสอบว่า v2 ไม่ทำลาย v1
3. ตรวจสอบ ethics compliance
4. วางแผน Phase 3 (ถ้ามี)

---

**Happy Testing! 🚀**
