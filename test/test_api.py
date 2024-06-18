from fastapi.testclient import TestClient
from api import app
client = TestClient(app)


def test_predict_price_xgboost():
    response = client.post("/predict_price/xgboost", json={
        "carat": 0.23,
        "cut": "Ideal",
        "color": "E",
        "clarity": "VS1",
        "depth": 61.5,
        "table": 55,
        "x": 3.95,
        "y": 3.98,
        "z": 2.43
    })
    assert response.status_code == 200
    assert "predicted_price" in response.json()


def test_predict_price_linear_regression():
    response = client.post("/predict_price/linear_regression", json={
        "carat": 0.23,
        "cut": "Ideal",
        "color": "E",
        "clarity": "VS1",
        "x": 3.95
    })
    assert response.status_code == 200
    assert "predicted_price" in response.json()


def test_get_similar_diamonds():
    response = client.post("/similar_diamonds", json={
        "carat": 0.23,
        "cut": "Ideal",
        "color": "E",
        "clarity": "VS1",
        "n": 5
    })
    assert response.status_code == 200
    similar_diamonds = response.json()
    assert isinstance(similar_diamonds, list)
    assert len(similar_diamonds) <= 5
