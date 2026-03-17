import joblib
import numpy as np

model = joblib.load("models/modelo_v1.joblib")

def predict(data):
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(features)

    return prediction.tolist()
'''

def predict(data):
    return {"resultado": "dummy"} 
'''
