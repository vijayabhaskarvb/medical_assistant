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

model = load_model("models/dr_model.keras")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def predict_dr(img_path):

    # ── 1. TensorFlow prediction (for confidence score) ──
    img = image.load_img(img_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_input = np.expand_dims(img_array, axis=0) / 255

    prediction = model.predict(img_input)[0][0]

    if prediction >= 0.5:
        tf_disease = "Normal Eye"
        confidence = prediction
    else:
        tf_disease = "Diabetic Retinopathy Detected"
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
You are an expert ophthalmologist analyzing a retinal fundus image for signs of Diabetic Retinopathy.

The AI diagnostic model has analyzed this image and found:
- Preliminary Result: {tf_disease}
- Model Confidence: {confidence_pct}%

Now carefully examine the retinal image yourself and provide a thorough medical report covering:

1. **Your Visual Diagnosis** - What do you observe in this retinal scan? Look for microaneurysms, hemorrhages, exudates, neovascularization. Do you agree with the preliminary result?
2. **Root Cause** - What underlying conditions are driving this? How does diabetes lead to this?
3. **Symptoms** - What vision-related symptoms would the patient likely be experiencing?
4. **Severity & Staging** - Rate the severity (No DR / Mild / Moderate / Severe / Proliferative DR) based on your observation.
5. **Treatment Options** - What treatments are appropriate at this stage?
6. **Medications** - What medications or injections are commonly used (e.g., Anti-VEGF, steroids)?
7. **Consultancy** - Which specialists should the patient urgently consult?
8. **Lifestyle Advice** - Key lifestyle changes to slow or prevent progression.
9. **Important Warning** - Any critical next steps the patient must take immediately.

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