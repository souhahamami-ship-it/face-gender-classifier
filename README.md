# FaceID — Gender Classifier Web App

A web app built around your `final_faces.ipynb` CNN notebook that classifies face images as **Male** or **Female**.

---

## Project Structure

```
gender-classifier/
├── index.html                   ← Frontend web app (works standalone)
├── app.py                       ← Flask backend (connects to your .pth model)
├── requirements.txt
├── gender_cnn_model_final.pth   ← Your trained model (copy here)
└── data/
    ├── men/
    └── women/
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Copy your trained model
```bash
cp path/to/gender_cnn_model_final.pth .
```

### 3. Start the Flask backend
```bash
python app.py
```

### 4. Open the web app
Open `index.html` in your browser — or serve it:
```bash
python -m http.server 8080
# then visit http://localhost:8080
```

---

## Connecting Frontend to Backend

In `index.html`, find the **`simulateInference`** function and replace it with **`realInference`**:

```javascript
// In the runAll() function, change:
await simulateInference(img);
// to:
const result = await realInference(img);
img.result = result;
updateCardUI(img);
```

The `realInference` function is already written and commented out at the bottom of the `<script>` tag:

```javascript
async function realInference(img) {
  const arrayBuffer = await img.file.arrayBuffer();
  const resp = await fetch('http://localhost:5000/predict', {
    method: 'POST',
    body: arrayBuffer,
    headers: { 'Content-Type': img.file.type }
  });
  return await resp.json(); // { label: 'men'|'women', confidence: 87.3 }
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server status check |
| POST | `/predict` | Single image (raw bytes in body) |
| POST | `/predict-batch` | Multiple images (multipart/form-data) |

### Example with curl
```bash
curl -X POST http://localhost:5000/predict \
  --data-binary @face.jpg \
  -H "Content-Type: image/jpeg"
```

### Response
```json
{
  "label": "women",
  "confidence": 91.4,
  "probabilities": {
    "men": 8.6,
    "women": 91.4
  }
}
```

---

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Custom 3-block CNN |
| Input | 224 × 224 RGB |
| Classes | men, women |
| Optimizer | Adam (lr=0.001) |
| Loss | CrossEntropyLoss |
| Dropout | 0.5 |
| Framework | PyTorch |

---
