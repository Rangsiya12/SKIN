# Skin Cancer Detection LINE Bot

โปรเจกต์นี้เป็น LINE Bot สำหรับตรวจจับโรคผิวหนังเบื้องต้นโดยใช้ YOLOv5

## การติดตั้ง

1. Clone repo นี้
2. ติดตั้ง dependencies: `pip install -r requirements.txt`
3. ตั้งค่า environment variables:
   - `LINE_CHANNEL_ACCESS_TOKEN`
   - `LINE_CHANNEL_SECRET`
4. วางไฟล์ model YOLOv5 ที่ `models/best.pt`
5. รันแอป: `python app.py`

## Deployment

แนะนำใช้ Render.com, Railway.app หรือ VPS ที่เปิดพอร์ต 5000