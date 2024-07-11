import dill
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import numpy as np

app = FastAPI()

# Загрузка модели и предварительных обработчиков с использованием dill
with open('rf_full_model.pkl', 'rb') as f:
    model = dill.load(f)
with open('onehot_encoder.pkl', 'rb') as f:
    encoder = dill.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = dill.load(f)

# Определение Pydantic модели для входных данных
class Form(BaseModel):
    visit_number: int
    utm_source: str
    utm_medium: str
    device_category: str
    device_os: str
    device_brand: str
    device_browser: str
    geo_country: str
    geo_city: str
    event_category: str
    short_model: str

# Определение Pydantic модели для предсказания
class Prediction(BaseModel):
    pred: str
    event_action: int

def preprocess_data(df):
    # Кодирование категориальных признаков
    categorical_features = ['utm_source', 'utm_medium', 'device_category', 
                            'device_os', 'device_brand', 'device_browser', 
                            'geo_country', 'geo_city', 'event_category', 'short_model']
    df_encoded = encoder.transform(df[categorical_features])
    
    # Нормализация числового признака
    numerical_features = ['visit_number']
    df_scaled = scaler.transform(df[numerical_features])
    
    # Преобразование в DataFrame
    df_scaled = pd.DataFrame(df_scaled, columns=['visit_number_std'])

    # Объединение закодированных и нормализованных признаков
    df_preprocessed = np.hstack([df_scaled, df_encoded])

    return df_preprocessed

@app.get('/status')
def status():
    status_message = "I'm OK"
    print("Status:", status_message)
    return {"status": status_message}

@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    print("Input data:", form.dict())
    df = pd.DataFrame.from_dict([form.dict()])
    print("DataFrame for prediction:", df)
    
    try:
        df_preprocessed = preprocess_data(df)
        print("Preprocessed DataFrame for prediction:", df_preprocessed)
    except Exception as e:
        print("Preprocessing error:", e)
        return {"error": str(e)}
    
    try:
        y = model.predict(df_preprocessed)
        print("Raw model prediction:", y)
    except Exception as e:
        print("Prediction error:", e)
        return {"error": str(e)}
    
    return {
        'pred': 'Prediction result',
        'event_action': int(y[0])  
    }