# Use Python 3.9 for TensorFlow and numpy compatibility
FROM python:3.9-slim

# Prevents Python from writing .pyc files to disc and enables stdout/stderr log flushing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN apt-get update && apt-get install -y gcc g++ libglib2.0-0 libsm6 libxext6 libxrender-dev && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Expose Gradio port
EXPOSE 7860

# Run app
CMD ["python", "app.py"]
