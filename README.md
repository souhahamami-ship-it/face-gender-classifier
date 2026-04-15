# Face Gender Classifier Web App

This project is a simple AI web application that classifies face images as **Male** or **Female** using a Convolutional Neural Network (CNN) built with PyTorch.

The system combines:

* A trained deep learning model
* A Python backend (Flask API)
* A frontend web interface (HTML + JavaScript)

---

## 🚀 What This Project Does

A user uploads an image in the browser →
The image is sent to a backend server →
The server runs the trained model →
The prediction is sent back and displayed on the page.

---

## 🧠 Model

* Built using a custom CNN (3 convolutional blocks)
* Trained in PyTorch (`final_faces.ipynb`)
* Input size: 224 × 224 RGB images
* Output: 2 classes → `men` or `women`
* Loss function: CrossEntropyLoss
* Optimizer: Adam

The trained model is saved as:

```
gender_cnn_model_final.pth
```

---

## 🏗️ Project Structure

```
gender-classifier/
├── index.html                 # Frontend (user interface)
├── app.py                     # Backend (Flask API)
├── requirements.txt           # Python dependencies
├── gender_cnn_model_final.pth # Trained model
└── data/                      # Training dataset
```

---

## ⚙️ How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Make sure the model file is in the project folder

```
gender_cnn_model_final.pth
```

### 3. Start the backend server

```bash
python app.py
```

This will start a local server at:

```
http://localhost:5000
```

### 4. Open the frontend

You can either:

* Open `index.html` directly
  or

```bash
python -m http.server 8080
```

Then go to:

```
http://localhost:8080
```

---

## 🔗 Connecting Frontend to Backend

By default, the frontend uses a **fake prediction function** (`simulateInference`) for testing.

To use the real model:

Find this line in `index.html`:

```javascript
await simulateInference(img);
```

Replace it with:

```javascript
const result = await realInference(img);
img.result = result;
updateCardUI(img);
```

### What this does

* Sends the image to the backend (`/predict`)
* The backend runs the model
* Returns a real prediction
* Displays the result in the UI

---

## 🌐 API Endpoints

### GET `/health`

Check if the server is running.

---

### POST `/predict`

Send a single image and get a prediction.

* Input: raw image bytes
* Output:

```json
{
  "label": "women",
  "confidence": 91.4
}
```

---

## 🔁 How the System Works (End-to-End)

1. User uploads an image in the browser
2. JavaScript converts the image into bytes
3. The image is sent to the Flask backend using a POST request
4. The backend:

   * loads the trained model
   * preprocesses the image
   * runs prediction
5. The backend returns a JSON response
6. The frontend displays the result

---

## 📌 Notes & Limitations

* The model only predicts two classes: `men` and `women`
* It assumes the input image already contains a face
* Performance depends on the training dataset

---

## 💬 About This Project

The CNN model and training pipeline were built from scratch.

The web interface and API structure were initially scaffolded with AI assistance, then reviewed and understood to connect the frontend and backend properly.

---
