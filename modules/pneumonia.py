import tensorflow as tf
import numpy as np
import cv2
import os
import base64
from tensorflow.keras.preprocessing import image
from groq import Groq
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

load_dotenv()

model = load_model("models/pneumonia_model.h5")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def predict_pneumonia(img_path):

    # ── 1. TensorFlow prediction (for confidence score) ──
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_input = np.expand_dims(img_array, axis=0) / 255

    prediction = model.predict(img_input)[0][0]

    if prediction >= 0.5:
        tf_disease = "Pneumonia Detected"
        confidence = prediction
    else:
        tf_disease = "Healthy Lung"
        confidence = 1 - prediction

    confidence_pct = round(float(confidence) * 100, 2)

    # ── 2. Saliency Heatmap ──
    img_tensor = tf.convert_to_tensor(img_input)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        loss = preds[:, 0]

    grads = tape.gradient(loss, img_tensor)
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()
    saliency = saliency / (np.max(saliency) + 1e-10)

    heatmap = cv2.resize(saliency, (400, 400))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.imread(img_path)
    original = cv2.resize(original, (400, 400))
    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    filename = os.path.basename(img_path)
    heatmap_filename = "heatmap_" + filename
    heatmap_path = os.path.join("uploads", heatmap_filename)
    cv2.imwrite(heatmap_path, superimposed)

    # ── 3. Groq Vision Analysis ──
    base64_image = encode_image(img_path)

    prompt = f"""
You are an expert pulmonologist and radiologist analyzing a chest X-ray for signs of Pneumonia.

The AI diagnostic model has analyzed this image and found:
- Preliminary Result: {tf_disease}
- Model Confidence: {confidence_pct}%

Now carefully examine the chest X-ray yourself and provide a thorough medical report covering:

1. **Your Visual Diagnosis** - What do you observe in this chest X-ray? Look for consolidation, infiltrates, opacity patterns. Do you agree with the preliminary result?
2. **Root Cause** - What is likely causing this? (Bacterial, Viral, Fungal?) What pathogens are commonly responsible?
3. **Symptoms** - What respiratory symptoms would the patient likely be experiencing?
4. **Severity** - How severe does this appear? (Mild / Moderate / Severe) Which lung regions are affected?
5. **Treatment Options** - What treatment protocol is recommended?
6. **Medications** - What antibiotics, antivirals, or supportive medications are typically prescribed?
7. **Consultancy** - Which specialist should the patient see and how urgently?
8. **Home Care & Recovery** - What should the patient do at home to aid recovery?
9. **Important Warning** - Any red-flag symptoms that require immediate emergency care.

Be clear, medically accurate, and structured. This report will be shown to the patient.
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )

    explanation = response.choices[0].message.content

    return tf_disease, confidence_pct, heatmap_filename, explanation