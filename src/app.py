"""Local chart upload interface. Codex-generated; see ATTRIBUTION.md."""

from io import BytesIO
import warnings
from flask import Flask, request, render_template_string, send_file
from PIL import Image, UnidentifiedImageError
import numpy as np
from .predict import Predictor
from .payoffs import CLASSES
from .generate import render, style
from .payoffs import sample_trade

PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Options Payoff Reader</title><style>
body{font:17px system-ui;background:#f2f5fa;color:#15243b;margin:0}main{max-width:860px;margin:40px auto;padding:28px;background:white;border-radius:18px}
h1{font-size:36px}p{line-height:1.6}.tag{color:#426184;font-size:13px;letter-spacing:2px}button{background:#184f86;color:white;border:0;border-radius:8px;padding:12px 20px;cursor:pointer}input{padding:14px 0}.result{background:#eaf3fd;padding:20px;border-radius:12px}.error{color:#a12626}a{color:#184f86}.examples{display:flex;gap:10px;flex-wrap:wrap}.examples a{padding:8px;background:#f0f4f8;border-radius:8px}small{color:#526078}img{max-width:100%}</style>
<main><p class="tag">EDUCATIONAL CHART RECOGNITION</p><h1>Options Payoff Reader</h1>
<p>Upload a profit/loss diagram at expiration to recognize one of eight supported shapes.</p>
<form method="post" enctype="multipart/form-data"><label for="chart">Chart image (PNG, JPEG or WebP; up to 8 MB)</label><br><input id="chart" name="chart" type="file" accept="image/png,image/jpeg,image/webp" required><br><button>Read chart</button></form>
{% if error %}<p role="alert" class="error">{{error}}</p>{% endif %}
{% if result %}<section class="result"><h2>{{result.category}}</h2><p>Model confidence: {{'%.1f'|format(result.confidence*100)}}%</p><p>{{result.explanation}}</p><small>Confidence is uncalibrated. A confident result can still be wrong.</small></section>{% endif %}
<h2>Try an example</h2><p>Each link runs a newly generated chart through the selected model.</p><div class="examples">{% for name in classes %}<a href="/?example={{loop.index0}}">{{name}}</a>{% endfor %}</div>
{% if example is not none %}<img src="/example/{{example}}.png" alt="Selected example profit and loss diagram">{% endif %}
<h2>What to expect</h2><p>Use a single complete curve with price increasing left to right and profit increasing upward. Show all strikes and both tails. The model was trained on synthetic charts; unfamiliar styles, screenshots, multiple curves and incomplete images can fail.</p>
<p>These are shape categories, not exact portfolio identification. Different positions can produce the same shape. This tool does not extract strikes or premiums, detect unsupported inputs, or provide trading advice.</p>
{% if cropped %}<p><strong>Layout requirement:</strong> This model uses a fixed plot-box crop. Uploads must match the example layout; automatic plot detection is not implemented.</p>{% endif %}
<small>Runs locally. Uploaded images are processed in memory and are not saved.</small></main></html>"""


def example_image(label):
    return render(
        sample_trade(label, np.random.default_rng(9000 + label)),
        style(np.random.default_rng(8000 + label)),
    )


def create_app(predictor=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    model = predictor or Predictor()

    def page(result=None, error=None, example=None, status=200):
        return (
            render_template_string(
                PAGE,
                result=result,
                error=error,
                classes=CLASSES,
                example=example,
                cropped=model.config.get("crop", False),
            ),
            status,
        )

    @app.errorhandler(413)
    def too_large(e):
        return page(error="Image exceeds the 8 MB upload limit.", status=413)

    @app.route("/", methods=["GET", "POST"])
    def home():
        if request.method == "GET":
            value = request.args.get("example")
            if value is None:
                return page()
            try:
                label = int(value)
                if label not in range(8):
                    raise ValueError()
                return page(result=model.predict(example_image(label)), example=label)
            except ValueError:
                return page(error="Choose one of the eight examples.", status=400)
        upload = request.files.get("chart")
        if upload is None or not upload.filename:
            return page(error="Choose an image first.", status=400)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                im = Image.open(upload.stream)
                if im.format not in ["PNG", "JPEG", "WEBP"]:
                    raise ValueError("Unsupported format. Use PNG, JPEG or WebP.")
                if im.width * im.height > 16000000 or min(im.size) < 32:
                    raise ValueError(
                        "Use an image at least 32 pixels per side and at most 16 million pixels."
                    )
                im.load()
                im = im.convert("RGB")
            return page(result=model.predict(im))
        except (
            UnidentifiedImageError,
            OSError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ):
            return page(
                error="Could not read this image. Please upload a valid PNG, JPEG or WebP.",
                status=400,
            )
        except ValueError as e:
            return page(error=str(e), status=400)

    @app.get("/example/<int:label>.png")
    def image(label):
        if label not in range(8):
            return "Unknown example", 404
        buf = BytesIO()
        example_image(label).save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5001, debug=False)
