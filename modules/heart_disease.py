import pickle
import numpy as np

data = pickle.load(open("models/heart_pipeline.pkl","rb"))

model = data["model"]
scaler = data["scaler"]

GENERAL_HEALTH = {
"Poor":0,"Fair":1,"Good":2,"Very good":3,"Excellent":4
}

CHECKUP = {
"Never":0,"Within 5 years":1,"Within 2 years":2,"Within 1 year":3
}

YES_NO = {"No":0,"Yes":1}

SEX = {"Female":0,"Male":1}

AGE = {
"18-24":0,"25-29":1,"30-34":2,"35-39":3,
"40-44":4,"45-49":5,"50-54":6,"55-59":7,
"60-64":8,"65-69":9,"70-74":10,"75-79":11,"80+":12
}

SMOKING = {"Never":0,"Former":1,"Current":2}


def predict_heart(form):

    input_data = np.array([[

        GENERAL_HEALTH[form["health"]],
        CHECKUP[form["checkup"]],
        YES_NO[form["exercise"]],
        YES_NO[form["depression"]],
        YES_NO[form["diabetes"]],
        YES_NO[form["arthritis"]],
        SEX[form["sex"]],
        AGE[form["age"]],
        float(form["bmi"]),
        SMOKING[form["smoking"]],
        float(form["alcohol"])

    ]])

    scaled = scaler.transform(input_data)

    prediction = model.predict(scaled)[0]

    if prediction == 1:
        return "Heart Disease Detected"
    else:
        return "No Heart Disease Detected"
