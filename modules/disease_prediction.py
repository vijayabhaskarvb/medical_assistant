import numpy as np
import pandas as pd
import pickle

# LOAD DATA
training = pd.read_csv("data/Training.csv")

X = training.drop('prognosis', axis=1)
y = training['prognosis']

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(y)

symptoms_dict = {symptom: idx for idx, symptom in enumerate(X.columns)}
diseases_list = le.inverse_transform(np.arange(len(le.classes_)))

svc = pickle.load(open("models/svc.pkl", "rb"))

sym_des = pd.read_csv("data/symtoms_df.csv")
precautions = pd.read_csv("data/precautions_df.csv")
workout = pd.read_csv("data/workout_df.csv")
description = pd.read_csv("data/description.csv")
medications = pd.read_csv("data/medications.csv")
diets = pd.read_csv("data/diets.csv")


def helper(dis):

    desc = " ".join(description.loc[description['Disease'] == dis, 'Description'].values)

    pre = precautions.loc[
        precautions['Disease'] == dis,
        ['Precaution_1','Precaution_2','Precaution_3','Precaution_4']
    ].values.flatten().tolist()

    med = medications.loc[medications['Disease'] == dis, 'Medication'].tolist()
    die = diets.loc[diets['Disease'] == dis, 'Diet'].tolist()
    wrk = workout.loc[workout['disease'] == dis, 'workout'].tolist()

    return desc, pre, med, die, wrk


def get_predicted_value(symptoms):

    input_vector = np.zeros(len(symptoms_dict))

    for s in symptoms:
        if s in symptoms_dict:
            input_vector[symptoms_dict[s]] = 1

    return diseases_list[svc.predict(input_vector.reshape(1,-1))[0]]


def predict_disease(user_input):

    symptoms = [s.strip() for s in user_input.split(",")]

    disease = get_predicted_value(symptoms)

    desc, pre, med, die, wrk = helper(disease)

    return {
        "disease": disease,
        "description": desc,
        "precautions": pre,
        "medications": med,
        "diets": die,
        "workout": wrk
    }
