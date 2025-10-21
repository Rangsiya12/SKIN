FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git gcc libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone YOLOv5 แล้วติดตั้ง dependencies ของมัน
RUN git clone https://github.com/ultralytics/yolov5 && \
    pip install -r yolov5/requirements.txt

# คัดลอกไฟล์ทั้งหมด (รวม app.py, models/)
COPY . .

# Railway ตั้ง ENV PORT อัตโนมัติ → ให้ bind พอร์ต 5000
ENV PORT=5000

CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]