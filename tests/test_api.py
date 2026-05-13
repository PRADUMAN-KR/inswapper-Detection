def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ready(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_detect(client, sample_image_bytes):
    response = client.post(
        "/detect",
        files={"file": ("sample.jpg", sample_image_bytes, "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["filename"] == "sample.jpg"
    assert body["result"]["label"] in {"real", "fake"}

