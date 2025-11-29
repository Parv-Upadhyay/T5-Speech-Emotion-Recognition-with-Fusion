# 1. Python Image Import karein
FROM python:3.9-slim

# 2. System Tools (FFmpeg aur OpenCV dependencies) install karein
# Cloud server par FFmpeg manual nahi daal sakte, isliye apt-get use karte hain
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 3. Working Directory set karein
WORKDIR /app

# 4. Requirements file copy karein
COPY requirements.txt .

# 5. Python Libraries install karein
RUN pip install --no-cache-dir -r requirements.txt

# 6. Baaki saari files copy karein
COPY . .

# 7. Permissions set karein (Hugging Face ke liye zaroori hai)
RUN chmod -R 777 /app

# 8. App Start karein
CMD ["python", "app.py"]