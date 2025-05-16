from fastapi.testclient import TestClient
from app.main import app
from PIL import Image
import io

client = TestClient(app)

def test_image_preprocess_endpoint():
    # Créer une image RGB simple (rouge)
    image = Image.new("RGB", (300, 300), color=(255, 0, 0))
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    response = client.post(
        "/preprocessing/image",
        files={"file": ("test.png", img_bytes, "image/png")}
    )

    assert response.status_code == 200
    result = response.json()
    assert "processed_image_base64" in result
    assert result["filename"] == "test.png"
    assert isinstance(result["processed_image_base64"], str)
    assert len(result["processed_image_base64"]) > 100  # base64 length check