import tensorflow as tf
import numpy as np
import cv2
import gradio as gr
import os
import gdown  # برای دانلود فایل از گوگل درایو

# لینک‌های فایل‌های مدل در گوگل درایو
model_h5_url = "https://drive.google.com/uc?id=1R6QYrLRAvkmYEPZxvApi7WvzXmL3Q8hU"
weights_h5_url = "https://drive.google.com/uc?id=1heNwBaKC3-ZpFdFP3MNt49MeX7sdnnHh"


# فولدر مدل‌ها
os.makedirs("model", exist_ok=True)
os.makedirs("weights", exist_ok=True)

def download_models():
    if not os.path.exists("model/model.h5"):
        print("📥 Downloading model.h5...")
        gdown.download(model_h5_url, "model/model.h5", quiet=False)
    if not os.path.exists("weights/modeldense1.h5"):
        print("📥 Downloading modeldense1.h5...")
        gdown.download(weights_h5_url, "weights/modeldense1.h5", quiet=False)

def load_model():
    download_models()
    model = tf.keras.models.load_model("model/model.h5")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        metrics=["accuracy"],
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    )
    model.load_weights("weights/modeldense1.h5")
    return model

model = load_model()

class_name = [
    'Benign with Density=1', 'Malignant with Density=1',
    'Benign with Density=2', 'Malignant with Density=2',
    'Benign with Density=3', 'Malignant with Density=3',
    'Benign with Density=4', 'Malignant with Density=4'
]

def preprocess(image):
    image = cv2.resize(image, (224, 224))
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    im = cv2.filter2D(image, -1, kernel)
    return im

def predict_img(img):
    img = preprocess(img)
    img = img / 255.0
    im = img.reshape(-1, 224, 224, 3)
    pred = model.predict(im)[0]
    return {class_name[i]: float(pred[i]) for i in range(len(class_name))}

iface = gr.Interface(fn=predict_img, inputs=gr.Image(type="numpy"), outputs=gr.Label(num_top_classes=8))
if __name__ == "__main__":
    iface.launch(debug=True, share=True)
