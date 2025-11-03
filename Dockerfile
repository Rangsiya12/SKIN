FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system-level dependencies (for OpenCV, torch, etc.)
RUN apt-get update && apt-get install -y \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy your full app (e.g. app.py, models/, detect_skin.py, etc.)
COPY . .

# Railway (or others) set PORT as env — default to 5000
ENV PORT=5000

# Launch Flask app via gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]