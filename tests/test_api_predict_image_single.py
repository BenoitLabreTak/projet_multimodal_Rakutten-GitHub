import pytest
from fastapi.testclient import TestClient
from app.main import app
import os

client = TestClient(app)

def test_predict_image_single():
    image_path = "data/images/test/image_55029630_product_1486851.jpg"
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    
    with open(image_path, "rb") as f:
        response = client.post(
            "/image/predict_image",  
            files={"file": ("image.jpg", f, "image/jpeg")}
        )

    assert response.status_code == 200
    assert "prediction" in response.json()