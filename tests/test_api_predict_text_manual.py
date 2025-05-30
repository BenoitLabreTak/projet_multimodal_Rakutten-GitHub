from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_text_form():
    response = client.post(
        "/predict/text/manual",
        data={"designation": "Stylo Rouge", "description": "Stylo bille pour prise de note"}
    )
    print(response.json())
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result