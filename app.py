from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# IMPORT MODULES
from modules.brain_tumor import predict_brain
from modules.pneumonia import predict_pneumonia
from modules.diabetic_retinopathy import predict_dr
from modules.heart_disease import predict_heart
from modules.disease_prediction import predict_disease
from modules.chatbot import ask_bot

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ================= DASHBOARD =================
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


# ================= SERVE UPLOADED FILES =================
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)


# ================= BRAIN =================
@app.route("/brain", methods=["GET","POST"])
def brain():

    result=None
    confidence=None
    explanation=None
    image=None
    heatmap=None

    if request.method=="POST":

        file=request.files["image"]

        path=os.path.join(UPLOAD_FOLDER,file.filename)
        file.save(path)

        result,confidence,heatmap_filename,explanation=predict_brain(path)

        image=file.filename
        heatmap=heatmap_filename

    return render_template(
        "brain.html",
        result=result,
        confidence=confidence,
        explanation=explanation,
        image=image,
        heatmap=heatmap
    )


# ================= PNEUMONIA =================
@app.route("/pneumonia", methods=["GET","POST"])
def pneumonia():

    result=None
    confidence=None
    explanation=None
    image=None
    heatmap=None

    if request.method=="POST":

        file=request.files["image"]

        path=os.path.join(UPLOAD_FOLDER,file.filename)
        file.save(path)

        result,confidence,heatmap_filename,explanation=predict_pneumonia(path)

        image=file.filename
        heatmap=heatmap_filename

    return render_template(
        "pneumonia.html",
        result=result,
        confidence=confidence,
        explanation=explanation,
        image=image,
        heatmap=heatmap
    )


# ================= DIABETIC RETINOPATHY =================
@app.route("/dr", methods=["GET","POST"])
def dr():

    result=None
    confidence=None
    explanation=None
    image=None
    heatmap=None

    if request.method=="POST":

        file=request.files["image"]

        path=os.path.join(UPLOAD_FOLDER,file.filename)
        file.save(path)

        result,confidence,heatmap_filename,explanation=predict_dr(path)

        image=file.filename
        heatmap=heatmap_filename

    return render_template(
        "dr.html",
        result=result,
        confidence=confidence,
        explanation=explanation,
        image=image,
        heatmap=heatmap
    )


# ================= HEART =================
@app.route("/heart", methods=["GET","POST"])
def heart():

    result=None

    if request.method=="POST":
        result=predict_heart(request.form)

    return render_template("heart.html",result=result)


# ================= DISEASE =================
@app.route("/disease", methods=["GET","POST"])
def disease():

    result=None

    if request.method=="POST":

        symptoms=request.form["symptoms"]

        result=predict_disease(symptoms)

    return render_template("disease.html",result=result)


# ================= CHATBOT PAGE =================
@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")


# ================= CHATBOT API =================
@app.route("/ask", methods=["POST"])
def ask():

    try:

        data=request.get_json()

        question=data.get("message", "")

        answer=ask_bot(question)

        return jsonify({"reply":answer})

    except Exception as e:

        print("CHATBOT ERROR:",e)

        return jsonify({"reply":"Sorry, chatbot is temporarily unavailable."})


# ================= RUN SERVER =================
if __name__=="__main__":
    app.run(debug=True)
