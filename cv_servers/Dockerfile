```dockerfile
# Windows-based Python image
FROM python:3.11-windowsservercore-ltsc2022

WORKDIR C:/app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8 weights during build
RUN python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"

# Copy project files
COPY . .

# Run your application
CMD ["python", "main.py"]
```
