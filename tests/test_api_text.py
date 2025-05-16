import pandas as pd
from io import BytesIO
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_text_preprocess_endpoint():
    response = client.post(
        "/preprocessing/text/manual",
        data={"designation": "Lego City", "description": "Camion de pompier avec échelle"}
    )

    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, dict)
    assert "cleaned_text" in result
    assert result["cleaned_text"] == "lego city camion de pompier avec echelle"