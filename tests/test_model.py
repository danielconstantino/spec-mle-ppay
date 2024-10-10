from fastapi.testclient import TestClient
from src.main import app
import io

client = TestClient(app)

def test_load_model():
    # Crie um arquivo de modelo fictício para o teste
    fake_model = io.BytesIO(b"fake model content")
    response = client.post("/model/load/", files={"file": ("fake_model.pkl", fake_model)})
    assert response.status_code == 200
    assert response.json() == {"message": "Modelo carregado com sucesso"}

def test_model_info():
    response = client.get("/model/info")
    assert response.status_code in [200, 404]  # 200 se o modelo estiver carregado, 404 caso contrário