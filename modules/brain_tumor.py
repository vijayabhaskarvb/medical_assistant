import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.preprocessing import image
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
load_dotenv()

model = load_model("models/brain_model.keras")

groq = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)

def predict_brain(img_path):

    img = image.load_img(img_path, target_size=(64,64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255

    prediction = model.predict(img)[0][0]

    if prediction >= 0.5:
        disease = "Healthy Brain"
        confidence = prediction
    else:
        disease = "Brain Tumor Detected"
        confidence = 1 - prediction

    img_tensor = tf.convert_to_tensor(img)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        preds = model(img_tensor)
        loss = preds[:,0]

    grads = tape.gradient(loss, img_tensor)

    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0]
    saliency = saliency.numpy()
    saliency = saliency / (np.max(saliency) + 1e-10)

    heatmap = cv2.resize(saliency, (400,400))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.imread(img_path)
    original = cv2.resize(original, (400,400))

    superimposed = cv2.addWeighted(original,0.6,heatmap,0.4,0)

    filename = os.path.basename(img_path)

    heatmap_filename = "heatmap_" + filename
    heatmap_path = os.path.join("uploads", heatmap_filename)

    cv2.imwrite(heatmap_path, superimposed)

    prompt = f"""
    Explain this medical condition.

    Condition: {disease}
    Confidence: {round(confidence*100,2)}%

    Explain:
    - Causes
    - Symptoms
    - Severity
    - Prevention
    - Treatment
    """

    response = groq.invoke([HumanMessage(content=prompt)])

    # RETURN ONLY THE FILE NAME
    return disease, round(confidence*100,2), heatmap_filename, response.content
