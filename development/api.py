from datetime import datetime
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from pymongo import MongoClient
from pipeline import linearl_model_dataprep

model_path_xgb = "models/XGBoost_Optuna_20240618_225725.pkl"
xgb_model = joblib.load(model_path_xgb)

model_path_linear = "models/LinearRegression_20240618_225725.pkl"
linear_model = joblib.load(model_path_linear)
diamonds = pd.read_csv("diamonds.csv")
x_train, x_test, y_train, y_test = linearl_model_dataprep(diamonds)
app = FastAPI()


def log_request(endpoint: str, request_data: dict):
    client = MongoClient(
        "mongodb+srv://hilaalurun:0aAFK6VxoDZsfbIB@hilal.pyqzfwe.mongodb.net/?retryWrites=true&w=majority&appName=Hilal")
    db = client['diamond_api_logs']
    collection = db['api_logs']
    log_entry = {
        'endpoint': endpoint,
        'request_data': request_data,
        'timestamp': datetime.now()
    }
    result = collection.insert_one(log_entry)
    return result.inserted_id


def log_response(endpoint: str, response_data: dict):
    client = MongoClient(
        "mongodb+srv://hilaalurun:0aAFK6VxoDZsfbIB@hilal.pyqzfwe.mongodb.net/?retryWrites=true&w=majority&appName=Hilal")
    db = client['diamond_api_logs']
    collection = db['api_logs']
    result = collection.find_one_and_update(
        {'endpoint': endpoint, 'response_data': None},
        {'$set': {'response_data': response_data}}
    )
    return result


class DiamondFeatures(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float


@app.post("/predict_price/xgboost")
def predict_price(features: DiamondFeatures):
    features = {
        "carat": [features.carat],
        "cut": [features.cut],
        "color": [features.color],
        "clarity": [features.clarity],
        "depth": [features.depth],
        "table": [features.table],
        "x": [features.x],
        "y": [features.y],
        "z": [features.z],
    }
    log_request('/predict_price/xgboost', features)
    new_diamond = pd.DataFrame(features)
    new_diamond['cut'] = pd.Categorical(new_diamond['cut'],
                                        categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'],
                                        ordered=True)
    new_diamond['color'] = pd.Categorical(new_diamond['color'],
                                          categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
    new_diamond['clarity'] = pd.Categorical(new_diamond['clarity'],
                                            categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2',
                                                        'I1'], ordered=True)
    prediction = xgb_model.predict(new_diamond)[0]
    log_response('/predict_price/xgboost', {'XGBoost': float(prediction)})
    return {"predicted_price": str(prediction)}


class DiamondFeatures_linear(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    x: float


@app.post("/predict_price/linear_regression")
def predict_price_linear(features: DiamondFeatures_linear):
    features = {
        "carat": [features.carat],
        "cut": [features.cut],
        "color": [features.color],
        "clarity": [features.clarity],
        "x": [features.x]
    }
    #log_request('/predict_price/linear_regression', features)
    new_diamond = pd.DataFrame(features)
    new_diamond_dummy = pd.get_dummies(new_diamond, columns=['cut', 'color', 'clarity'])
    template_columns = x_train.columns

    for col in template_columns:
        if col not in new_diamond_dummy.columns:
            new_diamond_dummy[col] = False

    new_diamond_dummy = new_diamond_dummy[template_columns]
    prediction = linear_model.predict(new_diamond_dummy)[0]
    #log_response('/predict_price/linear_regression', {'LinearRegression': float(prediction)})
    return {"predicted_price": str(prediction)}


class DiamondFeatures_prediction(BaseModel):
    carat: float
    cut: str
    color: str
    clarity: str
    n: int


@app.post("/similar_diamonds")
def get_similar_diamonds(features: DiamondFeatures_prediction):
    filtered_df = x_train[
        (x_train[f"cut_{features.cut}"] == f"cut_{features.cut}") &
        (x_train[f"color_{features.color}"] == f"color_{features.color}") &
        (x_train[f"clarity_{features.clarity}"] == f"clarity_{features.clarity}")
        ]
    #log_request('/similar_diamonds', features)
    filtered_df['carat_diff'] = (filtered_df['carat'] - features.carat).abs()
    similar_diamonds = filtered_df.sort_values('carat_diff').head(features.n)
    similar_diamonds = similar_diamonds.drop(columns=['carat_diff'])
    #log_response('/similar_diamonds', {'LinearRegression': similar_diamonds})
    return similar_diamonds


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)