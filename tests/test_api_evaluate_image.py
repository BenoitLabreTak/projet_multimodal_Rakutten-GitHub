import pandas as pd
from io import BytesIO
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_evaluate_image_file_api():
    # Exemple de DataFrame avec les colonnes attendues
    df = pd.DataFrame({
        "imageid": [1114982518],
        "productid": [1932975409],
        "prdtypecode": [1280]  # label ground truth
    })

    # Convertir en fichier CSV en mémoire
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)

    # Chemin vers les images de test
    image_dir = "data/images/test"

    # Appel à l'API
    response = client.post(
        "/evaluate/image/file",
        files={"file": ("test_image_eval.csv", buffer, "text/csv")},
        params={"sample_size": 1, "image_dir": image_dir}
    )

    # Vérification du statut de réponse
    assert response.status_code == 200

    # Optionnel : vérification du contenu retourné
    result = response.json()
    assert isinstance(result, dict)
    assert "average_confidence_score" in result
    assert "weighted_f1_score" in result