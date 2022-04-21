import numpy as np
import pandas as pd
from flask import Flask
from oauthlib.uri_validate import host
from sklearn.preprocessing import StandardScaler, LabelEncoder
import model
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pywebio import STATIC_PATH
from pywebio import *
from pywebio.input import *
from pywebio.output import *


# model1 = pickle.load(open('stroke.pkl', 'rb'))


def predict():
    put_grid([[None, put_text("Stroke prediction software"), None]], cell_width='350px', cell_height='50px')
    gender = select('Please select gender.', ['Male', 'Female'], required=True)
    if gender == 'Male':
        gender = 1
    elif gender == "Female":
        gender = 0

    age = input('Enter age', type=NUMBER, required=True)
    age = ((age - min(model.my_new_data2['age'])) / (max(model.my_new_data2['age']) - min(model.my_new_data2['age'])))

    hypertension = select('Do you have hypertension?', ['Yes', 'No'], required=True)
    if hypertension == 'Yes':
        hypertension = 1
    elif hypertension == 'No':
        hypertension = 0

    heart_disease = select('Do you have Heart Disease?', ['Yes', 'No'], required=True)
    if heart_disease == 'Yes':
        heart_disease = 1
    elif heart_disease == 'No':
        heart_disease = 0

    ever_married = select('Have you ever married?', ['Yes', 'No'], required=True)
    if ever_married == 'Yes':
        ever_married = 1
    elif ever_married == 'No':
        ever_married = 0

    work_type = select('what is your work type?', ['Government', 'Student', 'Private', 'Self-employed'], required=True)
    if work_type == 'Government':
        work_type = 0
    elif work_type == 'Student':
        work_type = 1
    elif work_type == 'Private':
        work_type = 2

    elif work_type == 'Self-employed':
        work_type = 3
    else:
        work_type = 4

    residence_type = select('What is your resident type?', ['Urban', 'Rural'], required=True)
    if residence_type == "Urban":
        residence_type = 1
    else:
        residence_type = 0

    avg_glucose_level = input('Enter Avg Glucose level', type=NUMBER, required=True)
    if avg_glucose_level == "i do not know":
        avg_glucose_level = np.mean(model.my_new_data2['avg_glucose_level'])
    else:
        avg_glucose_level = ((int(avg_glucose_level) - min(model.my_new_data2['avg_glucose_level'])) / (
                max(model.my_new_data2['avg_glucose_level']) - min(model.my_new_data2['avg_glucose_level'])))

    smoking_status = select('Select smoking status', ['Unknown', 'Never smoked', 'Formerly smoked', 'Smokes'],
                            required=True)
    if smoking_status == "Unknown":
        smoking_status = 0
    elif smoking_status == "Never smoked":
        smoking_status = 1
    elif smoking_status == "Formerly smoked":
        smoking_status = 2
    elif smoking_status == "Smokes":
        smoking_status = 3

    all_input = [gender, age, hypertension, heart_disease, ever_married, work_type, residence_type,
                 avg_glucose_level, smoking_status]

    # print(all_input)

    # load model
    model1 = pickle.load(open('stroke.pkl', 'rb'))
    # predictions
    result = model1.predict([all_input])

    if result == 1:
        put_text("Patient has chances of having Stroke")
    else:
        put_text("Patient has no risk of Stroke")

    return result


predict()
