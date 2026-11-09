"""Local chart upload interface. Codex-generated; see ATTRIBUTION.md."""

from io import BytesIO
import base64
import json
import warnings
from flask import Flask, request, render_template_string, send_file
from PIL import Image, ImageOps, UnidentifiedImageError
import numpy as np
from .predict import Predictor
from .payoffs import CLASSES
from .diverse import render_diverse
from .payoffs import sample_trade
from .measure import measure

PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Options Payoff Reader</title><style>
body{font:17px system-ui;background:#f2f5fa;color:#15243b;margin:0}main{max-width:860px;margin:40px auto;padding:28px;background:white;border-radius:18px}
h1{font-size:36px}p{line-height:1.6}.tag{color:#426184;font-size:13px;letter-spacing:2px}button{background:#184f86;color:white;border:0;border-radius:8px;padding:12px 20px;cursor:pointer}input{padding:14px 0}.result{background:#eaf3fd;padding:20px;border-radius:12px}.error{color:#a12626}a{color:#184f86}.examples{display:flex;gap:10px;flex-wrap:wrap}.examples a{padding:8px;background:#f0f4f8;border-radius:8px}small{color:#526078}img{max-width:100%}</style>
<main><p class="tag">EDUCATIONAL CHART RECOGNITION</p><h1>Options Payoff Reader</h1>
<p>Upload a profit/loss diagram at expiration to recognize one of eight supported shapes.</p>
<form method="post" enctype="multipart/form-data"><label for="chart">Chart image (PNG, JPEG or WebP; up to 8 MB)</label><br><input id="chart" name="chart" type="file" accept="image/png,image/jpeg,image/webp" required><br>
<label for="vertical-axis">What does the vertical axis show?</label> <select id="vertical-axis" name="vertical_axis"><option value="auto" {% if vertical_axis == 'auto' %}selected{% endif %}>Detect from labels</option><option value="profit_loss" {% if vertical_axis == 'profit_loss' %}selected{% endif %}>Profit/loss, including premium</option><option value="payoff" {% if vertical_axis == 'payoff' %}selected{% endif %}>Payoff, before premium</option></select><p><button>Read chart</button></p></form>
{% if error %}<p role="alert" class="error">{{error}}</p>{% endif %}
{% if result %}<section class="result"><h2>Suggested shape: {{result.category}}</h2><p>{{result.explanation}}</p><p>Read using {{result.source}}.</p><details><summary>Neural model comparison</summary><p>Neural prediction: {{result.neural_category}}. Model confidence: {{'%.1f'|format(result.confidence*100)}}%</p><small>Confidence is uncalibrated. A confident result can still be wrong.</small></details>
{% if result.uncertain %}<p><strong>This prediction is uncertain.</strong> Try a clearer image showing the full curve and both axes.</p>{% endif %}
{% if result.measurements %}<h3>Read values</h3><p>{{result.measurements.message}}</p>
{% if result.measurements.available %}{% set v=result.measurements['values'] %}<dl>
<dt>Estimated strikes (assuming this shape)</dt><dd>{{v.strikes|join(', ')}}</dd>
<dt>Estimated breakeven prices</dt><dd>{{v.breakevens|join(', ') if v.breakevens else 'Not established'}}</dd>
<dt>Estimated maximum profit</dt><dd>{{v.max_profit if v.max_profit is not none else 'Not established'}}</dd>
<dt>Estimated maximum loss</dt><dd>{{v.max_loss if v.max_loss is not none else 'Not established'}}</dd>
</dl><details><summary>Axis labels used for these estimates</summary><p>Price ticks: {% for tick in result.measurements.axis_evidence.x.ticks %}{{tick.value}}{% if tick.source == 'user' %} (confirmed by you){% elif tick.corrected_from %} (OCR: {{tick.corrected_from}}){% endif %}{% if not loop.last %}, {% endif %}{% endfor %}</p><p>Vertical ticks: {% if result.measurements.axis_evidence.y %}{% for tick in result.measurements.axis_evidence.y.ticks %}{{tick.value}}{% if tick.source == 'user' %} (confirmed by you){% endif %}{% if not loop.last %}, {% endif %}{% endfor %}{% else %}Not readable{% endif %}</p></details>{% endif %}{% endif %}</section>{% endif %}
{% if preview %}<h2>Your chart</h2><div id="chart-stage" style="position:relative;display:inline-block;max-width:100%"><img id="chart-preview" src="data:image/png;base64,{{preview}}" alt="Your uploaded chart; click a tick to confirm its position" style="display:block;cursor:crosshair"><div id="tick-marks" style="position:absolute;inset:0;pointer-events:none"></div></div>
<details><summary>Correct unreadable axis ticks</summary><p>For an axis you want to correct, enter a printed tick value, then click the corresponding <strong>tick mark</strong> on the chart. Repeat for a second, well-separated tick on that axis. The other axis can still be read automatically. Confirmed axes must be linear.</p>
<form method="post" id="calibration-form"><input type="hidden" name="chart_data" value="{{preview}}"><input type="hidden" id="axis-points" name="axis_points" value="[]">
<label for="tick-axis">Axis</label> <select id="tick-axis"><option value="x">Price (horizontal)</option><option value="y">Vertical</option></select>
<label for="tick-value">Printed value</label> <input id="tick-value" type="number" step="any" style="width:120px"><p id="tick-feedback" aria-live="polite">No ticks confirmed.</p>
<p><button type="button" id="clear-ticks">Clear ticks</button></p>
<label for="corrected-vertical-axis">Vertical axis</label> <select id="corrected-vertical-axis" name="vertical_axis"><option value="auto" {% if vertical_axis == 'auto' %}selected{% endif %}>Detect from labels</option><option value="profit_loss" {% if vertical_axis == 'profit_loss' %}selected{% endif %}>Profit/loss, including premium</option><option value="payoff" {% if vertical_axis == 'payoff' %}selected{% endif %}>Payoff, before premium</option></select>
<p><button>Read with these settings</button></p></form></details>
<script>
(() => {
let points = {{axis_points|tojson}};
const image=document.getElementById('chart-preview'), marks=document.getElementById('tick-marks'), field=document.getElementById('axis-points'), feedback=document.getElementById('tick-feedback');
function draw(){
 field.value=JSON.stringify(points); marks.replaceChildren();
 feedback.textContent=points.length ? points.map(p=>(p.axis==='x'?'Price':'Vertical')+': '+p.value).join('; ') : 'No ticks confirmed.';
 for(const p of points){const marker=document.createElement('span');marker.textContent='+';marker.style.cssText='position:absolute;font:bold 22px system-ui;transform:translate(-50%,-50%);color:'+(p.axis==='x'?'#1353ba':'#bf3030');marker.style.left=(p.position[0]*100)+'%';marker.style.top=(p.position[1]*100)+'%';marks.append(marker);}
}
image.addEventListener('click',event=>{
 const input=document.getElementById('tick-value');
 if(input.value.trim()==='' || !Number.isFinite(Number(input.value))){feedback.textContent='Enter the printed tick value before clicking its tick mark.';return;}
 if(points.length>=12){feedback.textContent='At most twelve ticks can be confirmed. Clear the ticks to start again.';return;}
 const rect=image.getBoundingClientRect();points.push({axis:document.getElementById('tick-axis').value,value:Number(input.value),position:[(event.clientX-rect.left)/rect.width,(event.clientY-rect.top)/rect.height]});draw();input.value='';
});
document.getElementById('clear-ticks').addEventListener('click',()=>{points=[];draw();});
document.getElementById('calibration-form').addEventListener('submit',event=>{for(const axis of ['x','y']){if(points.filter(p=>p.axis===axis).length===1){event.preventDefault();feedback.textContent='Confirm a second tick on the '+(axis==='x'?'price':'vertical')+' axis before reading.';return;}}});
draw();
})();
</script>{% endif %}
<h2>Try an example</h2><p>Each example demonstrates a different chart style.</p><div class="examples">{% for name in classes %}<a href="/?example={{loop.index0}}">{{name}}</a>{% endfor %}</div>
{% if example is not none %}<img src="/example/{{example}}.png" alt="Selected example profit and loss diagram">{% endif %}
<h2>What to expect</h2><p>Use a single complete curve with price increasing left to right and profit increasing upward. Show all strikes and both tails. The model was trained on synthetic charts; unfamiliar styles, screenshots, multiple curves and incomplete images can fail.</p>
<p>Different positions can produce the same shape. Numerical estimates require readable, linear axes and a single clear curve. Missing information is left unreadable. Unsupported charts may still receive a suggested shape.</p>
{% if cropped %}<p><strong>Layout requirement:</strong> This model uses a fixed plot-box crop. Uploads must match the example layout; automatic plot detection is not implemented.</p>{% endif %}
<small>Runs locally with no API fees. Uploaded images are processed in memory and are not saved or sent to an external service.</small></main></html>"""


def example_image(label):
    return render_diverse(
        sample_trade(label, np.random.default_rng(9000 + label)),
        (2 + label * 7) % 50,
        8000 + label,
        (800, 600),
    )


def create_app(predictor=None):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
    app.config["MAX_FORM_MEMORY_SIZE"] = 8 * 1024 * 1024
    model = predictor or Predictor()

    def read_chart(im, vertical_axis="auto", axis_points=None):
        if np.asarray(im.convert("L")).std() < 2:
            raise ValueError("No readable chart found. Upload an image with a visible curve and axes.")
        result = model.predict(im)
        result["uncertain"] = result.get("uncertain", result["confidence"] < .8)
        result["measurements"] = (
            dict(available=False, values=None, message="Shape prediction is uncertain; numerical estimates are withheld.")
            if result["uncertain"] else measure(im, CLASSES.index(result["category"]), vertical_axis=vertical_axis, axis_points=axis_points)
        )
        return result

    def page(result=None, error=None, example=None, status=200, preview=None, vertical_axis="auto", axis_points=None):
        return (
            render_template_string(
                PAGE,
                result=result,
                error=error,
                classes=CLASSES,
                example=example,
                cropped=model.config.get("crop", False),
                preview=preview,
                vertical_axis=vertical_axis,
                axis_points=axis_points or [],
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
                return page(result=read_chart(example_image(label)), example=label)
            except ValueError:
                return page(error="Choose one of the eight examples.", status=400)
        upload = request.files.get("chart")
        encoded = request.form.get("chart_data")
        if (upload is None or not upload.filename) and not encoded:
            return page(error="Choose an image first.", status=400)
        try:
            vertical_axis=request.form.get("vertical_axis","auto")
            if vertical_axis not in ("auto","profit_loss","payoff"):
                raise ValueError("Choose a valid vertical-axis interpretation.")
            axis_points=json.loads(request.form.get("axis_points","[]"))
            from .calibration import manual_axes
            manual_axes(axis_points)
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                stream=BytesIO(base64.b64decode(encoded,validate=True)) if encoded else upload.stream
                im = Image.open(stream)
                if im.format not in ["PNG", "JPEG", "WEBP"]:
                    raise ValueError("Unsupported format. Use PNG, JPEG or WebP.")
                if im.width * im.height > 16000000 or min(im.size) < 32:
                    raise ValueError(
                        "Use an image at least 32 pixels per side and at most 16 million pixels."
                    )
                im.load()
                im = ImageOps.exif_transpose(im)
                if im.mode in ("RGBA", "LA") or "transparency" in im.info:
                    rgba = im.convert("RGBA")
                    im = Image.new("RGBA", im.size, "white")
                    im.alpha_composite(rgba)
                im = im.convert("RGB")
            result=read_chart(im,vertical_axis,axis_points)
            thumbnail=im.copy()
            thumbnail.thumbnail((1200,1200))
            buf=BytesIO()
            thumbnail.save(buf,format="PNG")
            preview=base64.b64encode(buf.getvalue()).decode("ascii")
            return page(result=result,preview=preview,vertical_axis=vertical_axis,axis_points=axis_points)
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
