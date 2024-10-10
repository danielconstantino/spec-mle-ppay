from fastapi.testclient import TestClient
from src.main import app
from src.models import FlightInfo

client = TestClient(app)

def test_predict_delay():
    flight_info = FlightInfo(
        origin="JFK",
        destination="LAX",
        departure_time="2023-05-01T10:00:00",
        airline="Delta",
        flight_number="DL123"
    )
    response = client.post("/model/predict/", json=flight_info.dict())
    assert response.status_code == 200
    assert "delay_prediction" in response.json()

def test_predict_delay_invalid_input():
    invalid_flight_info = {"invalid_field": "value"}
    response = client.post("/model/predict/", json=invalid_flight_info)
    assert response.status_code == 422  # Unprocessable Entity