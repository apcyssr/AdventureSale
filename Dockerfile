ฃ# 1. ใช้ Python เวอร์ชันมาตรฐาน
FROM python:3.10-slim

# 2. ตั้งค่าโฟลเดอร์ทำงานในตัวเซิร์ฟเวอร์
WORKDIR /app

# 3. ก๊อปปี้ไฟล์ requirements.txt เข้าไปเพื่อติดตั้ง library
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. ก๊อปปี้โค้ดทั้งหมดในโปรเจกต์เข้าไปในเซิร์ฟเวอร์
COPY . .

# 5. เปิดพอร์ต 7860 (Hugging Face Spaces บังคับให้ใช้พอร์ตนี้เท่านั้น)
EXPOSE 7860

# 6. คำสั่งรัน Flask App โดยผูกกับพอร์ต 7860
CMD ["python", "app.py"]