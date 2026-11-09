"""Real selected-model upload smoke checks. Codex-generated."""

from io import BytesIO
import base64
import json
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


def test_numeric_values_are_visible(client, monkeypatch):
    monkeypatch.setattr("src.app.measure", lambda *args, **kwargs: dict(
        available=True, message="Approximate values", values=dict(
            strikes=[100.25], breakevens=[105.5], max_profit="unbounded", max_loss=5.25),
        axis_evidence=dict(x=dict(ticks=[]), y=None)))
    response=client.get("/?example=0")
    assert response.status_code==200
    for value in [b"100.25",b"105.5",b"unbounded",b"5.25"]:
        assert value in response.data


def test_confirmed_ticks_round_trip_without_saving_upload(client,monkeypatch):
    received=[]
    def fake_measure(image,label,**kwargs):
        received.append(kwargs)
        return dict(available=False,values=None,message="Reviewed axis settings")
    monkeypatch.setattr("src.app.measure",fake_measure)
    image=BytesIO();example_image(0).save(image,"PNG")
    points=[dict(axis="x",position=[.2,.8],value=80),dict(axis="x",position=[.8,.8],value=120)]
    response=client.post("/",data=dict(chart_data=base64.b64encode(image.getvalue()).decode(),axis_points=json.dumps(points),vertical_axis="profit_loss"))
    assert response.status_code==200
    assert received[0]["axis_points"]==points
    assert received[0]["vertical_axis"]=="profit_loss"
    assert b'id="chart-preview"' in response.data
    assert client.post("/",data=dict(chart_data=base64.b64encode(image.getvalue()).decode(),axis_points='{"invalid":true}')).status_code==400
