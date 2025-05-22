import os
from fastapi.testclient import TestClient
from app.main import app

base_url = os.environ.get("BASE_URL", "http://localhost:8000")
client = TestClient(app, base_url=base_url)