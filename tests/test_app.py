import requests

def test_homepage():
    response = requests.get("http://localhost:8000/")
    assert response.status_code == 200
    assert "<form" in response.text  # confirm HTML form is present

def test_generate_emojis():
    response = requests.post(
        "http://localhost:8000/generate",
        data={"num_images": 2}
    )
    assert response.status_code == 200
    assert "static/generated/" in response.text  # confirm images are being served
