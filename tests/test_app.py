"""Real selected-model upload smoke checks. Codex-generated."""

from io import BytesIO
import pytest
from PIL import Image
from src.app import create_app, example_image


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


def test_home_and_examples(client):
    assert client.get("/").status_code == 200
    for i in range(8):
        response = client.get(f"/?example={i}")
        assert response.status_code == 200 and b"Model confidence:" in response.data
        assert client.get(f"/example/{i}.png").mimetype == "image/png"


def test_upload_and_invalid_inputs(client):
    b = BytesIO()
    example_image(0).save(b, format="PNG")
    b.seek(0)
    r = client.post("/", data={"chart": (b, "chart.png")})
    assert r.status_code == 200 and b"Model confidence:" in r.data
    assert (
        client.post(
            "/", data={"chart": (BytesIO(b"not image"), "fake.png")}
        ).status_code
        == 400
    )
    assert client.post("/").status_code == 400
    assert client.get("/?example=99").status_code == 400
    tiny = BytesIO()
    Image.new("RGB", (2, 2)).save(tiny, format="PNG")
    tiny.seek(0)
    assert client.post("/", data={"chart": (tiny, "tiny.png")}).status_code == 400
    assert (
        client.post(
            "/", data={"chart": (BytesIO(b"x" * 9_000_000), "large.png")}
        ).status_code
        == 413
    )
