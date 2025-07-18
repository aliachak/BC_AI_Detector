import tensorflow as tf
import numpy as np
import cv2
import gradio as gr
import os
import gdown
import logging
import time

# تنظیم لاگ برای دیباگ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# لینک‌های فایل‌های مدل در گوگل درایو
model_h5_url = "https://drive.google.com/uc?id=1R6QYrLRAvkmYEPZxvApi7WvzXmL3Q8hU"
weights_h5_url = "https://drive.google.com/uc?id=1heNwBaKC3-ZpFdFP3MNt49MeX7sdnnHh"

# فولدر مدل‌ها
os.makedirs("model", exist_ok=True)
os.makedirs("weights", exist_ok=True)

def download_models():
    try:
        # بررسی وجود فایل‌ها
        if not os.path.exists("model/model.h5"):
            logger.info("📥 Downloading model.h5...")
            gdown.download(model_h5_url, "model/model.h5", quiet=False, resume=True)
        else:
            logger.info("model.h5 already exists, skipping download.")
        
        if not os.path.exists("weights/modeldense1.h5"):
            logger.info("📥 Downloading modeldense1.h5...")
            gdown.download(weights_h5_url, "weights/modeldense1.h5", quiet=False, resume=True)
        else:
            logger.info("modeldense1.h5 already exists, skipping download.")
    except Exception as e:
        logger.error(f"Failed to download models: {e}")
        raise

def load_model():
    try:
        download_models()
        logger.info("Loading model...")
        model = tf.keras.models.load_model("model/model.h5")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            metrics=["accuracy"],
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
        )
        model.load_weights("weights/modeldense1.h5")
        logger.info("Model loaded successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

# لود مدل با تأخیر برای اطمینان از دانلود کامل
model = None
try:
    model = load_model()
except Exception as e:
    logger.error(f"Model loading failed: {e}")
    exit(1)

class_name = [
    'Benign with Density=1', 'Malignant with Density=1',
    'Benign with Density=2', 'Malignant with Density=2',
    'Benign with Density=3', 'Malignant with Density=3',
    'Benign with Density=4', 'Malignant with Density=4'
]

def preprocess(image):
    try:
        image = cv2.resize(image, (224, 224))
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        im = cv2.filter2D(image, -1, kernel)
        return im
    except Exception as e:
        logger.error(f"Error in preprocessing image: {e}")
        raise

def predict_img(img):
    try:
        img = preprocess(img)
        img = img / 255.0
        im = img.reshape(-1, 224, 224, 3)
        pred = model.predict(im)[0]
        return {class_name[i]: float(pred[i]) for i in range(len(class_name))}
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise

iface = gr.Interface(fn=predict_img, inputs=gr.Image(type="numpy"), outputs=gr.Label(num_top_classes=8))

if __name__ == "__main__":
    # استفاده از پورت 7860 برای سازگاری با render.yaml
    port = int(os.environ.get('PORT', 7860))
    logger.info(f"Starting Gradio on port {port}")
    iface.launch(server_name="0.0.0.0", server_port=port)
