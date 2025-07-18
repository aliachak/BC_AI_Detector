FROM python:3.11

# جلوگیری از cache غیر ضروری
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# نصب وابستگی‌های سیستمی
RUN apt-get update && apt-get install -y \
    libpng-dev \
    libjpeg-dev \
    libopenjp2-7 \
    libtiff5 \
    zlib1g-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# نصب وابستگی‌ها
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# کپی کد پروژه
COPY . .

# پورت Gradio
EXPOSE 7860

# اجرای اپ
CMD ["python", "app.py"]
