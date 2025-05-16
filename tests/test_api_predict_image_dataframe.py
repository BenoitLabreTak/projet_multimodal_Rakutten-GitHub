import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_image_file():
    with open("data/text/test.csv", "rb") as f:
        response = client.post("/image/predict_dataframe", files={"file": ("test.csv", f, "text/csv")})
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list)
    assert "predicted_label" in result[0]