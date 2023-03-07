import numpy as np
from pywebio import *
from pywebio.input import *
from pywebio.output import *
from pywebio.platform.flask import start_server
from pywebio.session import run_js

import joblib


# define a function to handle undefined keys
def handle_error():
    put_text('Invalid input!')


# dictionary to map inputs to functions
dispatch = {
    'gender': lambda: select('Please select gender.', ['Male', 'Female'], required=True),
    'age': lambda: input('Enter age', type=NUMBER, required=True),
    'hypertension': lambda: select('Do you have hypertension?', ['Yes', 'No'], required=True),
    'heart_disease': lambda: select('Do you have Heart Disease?', ['Yes', 'No'], required=True),
    'ever_married': lambda: select('Have you ever married?', ['Yes', 'No'], required=True),
    'work_type': lambda: select('what is your work type?', ['Government', 'Student', 'Private', 'Self-employed'],
                                required=True),
    'residence_type': lambda: select('What is your resident type?', ['Urban', 'Rural'], required=True),
    'avg_glucose_level': lambda: input('Enter Avg Glucose level', type=NUMBER, required=True),
    'smoking_status': lambda: select('Select smoking status', ['Unknown', 'Never smoked', 'Formerly smoked', 'Smokes'],
                                     required=True)
}

# Add a default case to handle undefined keys
dispatch.setdefault(None, handle_error)


def predict():
    put_grid([[None, put_text("Stroke prediction software"), None]], cell_width='350px', cell_height='50px')

    # get input values using the dispatch dictionary
    gender = dispatch['gender']()
    gender = 1 if gender == 'Male' else 0

    age = dispatch['age']()

    hypertension = dispatch['hypertension']()
    hypertension = 1 if hypertension == 'Yes' else 0

    heart_disease = dispatch['heart_disease']()
    heart_disease = 1 if heart_disease == 'Yes' else 0

    ever_married = dispatch['ever_married']()
    ever_married = 1 if ever_married == 'Yes' else 0

    work_type = dispatch['work_type']()
    work_type = ['Government', 'Student', 'Private', 'Self-employed'].index(work_type)

    residence_type = dispatch['residence_type']()
    residence_type = 1 if residence_type == "Urban" else 0

    avg_glucose_level = dispatch['avg_glucose_level']()

    smoking_status = dispatch['smoking_status']()
    smoking_status = ['Unknown', 'Never smoked', 'Formerly smoked', 'Smokes'].index(smoking_status)

    # load model
    model1 = joblib.load('final_model.sav')

    # make prediction
    patient_data = np.array(
        [gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level,
         smoking_status]).reshape(1, -1)
    pred_stroke = model1.predict(patient_data)

    # show prediction result
    if pred_stroke == 1:
        put_text("Patient has chances of having Stroke.")
    elif pred_stroke == 0:
        put_text("Patient has no risk of Stroke.")

    # add restart button
    def restart():
        # clear previous inputs
        for key in dispatch.keys():
            del dispatch[key]
        predict()

    put_button("Restart", onclick=lambda: run_js('window.location.reload()'))


if __name__ == '__main__':
    start_server(predict, port=8080, debug=True)

predict()
