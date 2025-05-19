from fastapi.testclient import TestClient
from app.main import app
client = TestClient(app, base_url="http://localhost:8000")