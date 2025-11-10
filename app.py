import os
import cv2
import numpy as np
import onnxruntime as ort
import requests
from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename
from datetime import datetime
from googletrans import Translator
from dotenv import load_dotenv

# ----------------------------
# Setup
# ----------------------------
load_dotenv()

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
MODEL_PATH = "best.onnx"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["RESULT_FOLDER"] = RESULT_FOLDER

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# Languages for translation dropdown
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "gu": "Gujarati",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali"
}
translator = Translator()

# ----------------------------
# Helpers
# ----------------------------
def preprocess_image(image_path, target_size=640):
    orig = cv2.imread(image_path)
    if orig is None:
        raise ValueError(f"Unable to read image: {image_path}")
    h, w = orig.shape[:2]
    rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (target_size, target_size))
    inp = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(inp, 0), orig, {"orig_w": w, "orig_h": h, "target_size": target_size}


def decode_predictions(preds, meta, conf_thresh=0.25, iou_thresh=0.45):
    arr = np.asarray(preds)
    if arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[1] == 5:
        arr = arr.squeeze(0).T
    elif arr.ndim == 3 and arr.shape[0] == 1 and arr.shape[2] == 5:
        arr = arr.squeeze(0)
    elif arr.ndim == 2 and arr.shape[1] == 5:
        pass
    else:
        arr = arr.reshape(-1, 5)

    boxes, scores = [], []
    w, h, t = meta["orig_w"], meta["orig_h"], meta["target_size"]

    for p in arr:
        if len(p) < 5:
            continue
        cx, cy, bw, bh, conf = p[:5]
        if conf < conf_thresh:
            continue
        if 0 <= cx <= 1 and 0 <= cy <= 1:
            cx, cy, bw, bh = cx * t, cy * t, bw * t, bh * t
        x1, y1, x2, y2 = int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)
        boxes.append([x1, y1, x2, y2])
        scores.append(float(conf))

    if not boxes:
        return np.empty((0, 4), dtype=int), np.array([])

    boxes = np.array(boxes)
    sx, sy = w / t, h / t
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * sx
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * sy
    boxes = boxes.astype(int)

    keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores, conf_thresh, iou_thresh)
    if len(keep) == 0:
        return np.empty((0, 4), dtype=int), np.array([])
    keep = np.array(keep).reshape(-1)
    return boxes[keep], np.array(scores)[keep]


def draw_boxes(img, boxes, scores):
    out = img.copy()
    for (x1, y1, x2, y2), conf in zip(boxes, scores):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(out, f"Lesion {conf:.2f}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    return out


def generate_local_summary(lesion_count, avg_conf):
    if lesion_count == 0:
        return "No lesions detected — the retina appears normal in this scan."
    if avg_conf < 0.4:
        return ("We've looked at your eye screening results. The system noticed one very small area that might be "
                "something to watch, but it wasn’t very sure. This usually isn’t serious — just a minor spot to keep an eye on.")
    else:
        return ("The AI detected possible signs of diabetic retinopathy with moderate confidence. "
                "We recommend you consult an ophthalmologist for a full evaluation.")


def translate_summary(summary):
    results = {"en": summary}
    for code in LANGUAGES:
        if code == "en":
            continue
        try:
            t = translator.translate(summary, dest=code)
            results[code] = t.text
        except Exception:
            results[code] = summary
    return results


def get_client_city(request):
    """Fallback city for local testing."""
    try:
        ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        if ip in ("127.0.0.1", "::1", None):
            return "Ahmedabad"
        res = requests.get(f"https://ipinfo.io/{ip}/json", timeout=5)
        city = res.json().get("city", "Ahmedabad")
        return city
    except Exception:
        return "Ahmedabad"


def find_doctors(city):
    try:
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": f"ophthalmologist {city}", "format": "json", "limit": 4}
        headers = {"User-Agent": "NeuroLensAI/1.0"}
        res = requests.get(url, params=params, headers=headers, timeout=6)
        data = res.json()
        return [{
            "name": d.get("name", "Ophthalmologist"),
            "address": d.get("display_name", ""),
            "link": f"https://www.openstreetmap.org/?mlat={d['lat']}&mlon={d['lon']}#map=18/{d['lat']}/{d['lon']}"
        } for d in data]
    except Exception:
        return []


def run_inference(image_path, filename):
    img_input, orig, meta = preprocess_image(image_path)
    preds = session.run(None, {input_name: img_input})[0]
    boxes, scores = decode_predictions(preds, meta)
    result = draw_boxes(orig, boxes, scores)
    save_path = os.path.join(RESULT_FOLDER, filename)
    cv2.imwrite(save_path, result)
    lesion_count = len(scores)
    avg_conf = float(np.mean(scores)) if lesion_count else 0.0
    summary = generate_local_summary(lesion_count, avg_conf)
    translations = translate_summary(summary)
    return save_path, summary, translations, lesion_count, avg_conf


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html", logo_url=url_for("static", filename="assets/logo.png"))


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return redirect(url_for("index"))
    file = request.files["file"]
    if file.filename == "":
        return redirect(url_for("index"))
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    result_path, summary, translations, lesion_count, avg_conf = run_inference(path, filename)
    city = get_client_city(request)
    doctors = find_doctors(city)

    return render_template(
        "result.html",
        result_image=url_for("static", filename=f"results/{filename}"),
        summary_translations=translations,
        languages=LANGUAGES,
        lesion_count=lesion_count,
        avg_conf=avg_conf,
        city=city,
        doctors=doctors,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        logo_url=url_for("static", filename="assets/logo.png"),
    )


if __name__ == "__main__":
    app.run(debug=True)
