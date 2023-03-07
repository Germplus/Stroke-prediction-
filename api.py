from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from starlette.responses import JSONResponse

app = FastAPI()

# load model
model = joblib.load('final_model.sav')

class PatientData(BaseModel):
    gender: str
    age: float
    hypertension: str
    heart_disease: str
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    smoking_status: str

@app.post("/predict")
async def predict(patient_data: PatientData):
    # extract input values
    gender = 1 if patient_data.gender.lower() == 'male' else 0
    age = patient_data.age
    hypertension = 1 if patient_data.hypertension.lower() == 'yes' else 0
    heart_disease = 1 if patient_data.heart_disease.lower() == 'yes' else 0
    ever_married = 1 if patient_data.ever_married.lower() == 'yes' else 0
    work_type = ['government', 'student', 'private', 'self-employed']
    if patient_data.work_type.lower() not in work_type:
        raise HTTPException(status_code=422, detail="Invalid work type")
    work_type = work_type.index(patient_data.work_type.lower())
    residence_type = 1 if patient_data.residence_type.lower() == 'urban' else 0
    avg_glucose_level = patient_data.avg_glucose_level
    smoking_status = ['unknown', 'never smoked', 'formerly smoked', 'smokes']
    if patient_data.smoking_status.lower() not in smoking_status:
        raise HTTPException(status_code=422, detail="Invalid smoking status")
    smoking_status = smoking_status.index(patient_data.smoking_status.lower())

    # make prediction
    patient_data = np.array(
        [gender, age, hypertension, heart_disease, ever_married, work_type, residence_type, avg_glucose_level,
         smoking_status]).reshape(1, -1)
    pred_stroke = model.predict(patient_data)

    # return prediction
    return {"stroke_prediction": "Yes" if pred_stroke == 1 else "No"}

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

@app.exception_handler(Exception)
async def exception_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
