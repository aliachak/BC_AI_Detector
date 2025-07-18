FROM python:3.10-slim

# جلوگیری از cache غیر ضروری
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# نصب وابستگی‌ها
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# کد پروژه
COPY . .

# پورت Gradio
EXPOSE 7860

# اجرای اپ
CMD ["python", "app.py"]
