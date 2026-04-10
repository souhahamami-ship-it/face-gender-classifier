"""
app.py — Flask backend for FaceID Gender Classifier
Run: python app.py
Then open index.html in your browser (or serve with 'python -m http.server 8080')
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from PIL import Image
import io

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR, static_url_path="")
CORS(app)

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

# ─── Model definition (must match your training code) ───────────────────────
class GenderCNN(nn.Module):
    def __init__(self):
        super(GenderCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 28 * 28, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2   = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 224 → 112
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 112 → 56
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 56  → 28
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

# ─── Preprocessing (same as test_transforms in the notebook) ─────────────────
test_transforms = v2.Compose([
    v2.Resize((224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─── Load model ──────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LABELS = ["men", "women"]

print(f"Loading model on {DEVICE}...")
model = GenderCNN().to(DEVICE)
model.load_state_dict(torch.load("gender_cnn_model_final.pth", map_location=DEVICE))
model.eval()
print("Model ready!")

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "device": str(DEVICE)})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts raw image bytes in request body.
    Returns: { label: 'men'|'women', confidence: 87.3, probabilities: { men: 87.3, women: 12.7 } }
    """
    try:
        # Read image from request body
        img_bytes = request.data
        if not img_bytes:
            return jsonify({"error": "No image data received"}), 400

        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        tensor = test_transforms(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_idx = torch.argmax(probs).item()

        male_pct   = round(probs[0].item() * 100, 1)
        female_pct = round(probs[1].item() * 100, 1)

        return jsonify({
            "label": LABELS[pred_idx],
            "confidence": round(probs[pred_idx].item() * 100, 1),
            "probabilities": {
                "men": male_pct,
                "women": female_pct
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict-batch", methods=["POST"])
def predict_batch():
    """
    Accepts multipart/form-data with multiple 'images' fields.
    Returns list of predictions.
    """
    results = []
    files = request.files.getlist("images")

    for f in files:
        try:
            img = Image.open(f.stream).convert("RGB")
            tensor = test_transforms(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                output = model(tensor)
                probs = torch.softmax(output, dim=1)[0]
                pred_idx = torch.argmax(probs).item()

            results.append({
                "filename": f.filename,
                "label": LABELS[pred_idx],
                "confidence": round(probs[pred_idx].item() * 100, 1),
                "probabilities": {
                    "men": round(probs[0].item() * 100, 1),
                    "women": round(probs[1].item() * 100, 1)
                }
            })
        except Exception as e:
            results.append({"filename": f.filename, "error": str(e)})

    return jsonify({"results": results, "total": len(results)})


if __name__ == "__main__":
    print("\n🚀 FaceID server starting at http://localhost:5000")
    print("   Open index.html in your browser to use the web app\n")
    app.run(debug=False, host="0.0.0.0", port=5000)