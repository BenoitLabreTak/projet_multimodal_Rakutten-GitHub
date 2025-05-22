import pandas as pd
from io import BytesIO
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_evaluate_text_api_file():
    # Create a sample CSV with necessary columns
    df = pd.DataFrame({
        "designation": ["Avenue Mandarine Jeux De Cartes Dinoptura"],
        "description": [""],
        "prdtypecode": [1281]  # Ground truth labels
    })

    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    response = client.post(
        "/evaluate/text/file",
        files={"file": ("data/text/test.csv", buffer, "text/csv")}
    )

    assert response.status_code == 200
    result = response.json()

    # Ensure keys are present
    assert "average_confidence_score" in result
    assert "weighted_f1_score" in result

    # Ensure values are floats and in expected range
    assert isinstance(result["average_confidence_score"], float)
    assert isinstance(result["weighted_f1_score"], float)
    assert 0.0 <= result["average_confidence_score"] <= 1.0
    assert 0.0 <= result["weighted_f1_score"] <= 1.0
