# Instagram Report Generator — Text-Anchor OCR Edition

A sleek, modern single-page web app and Python backend service that generates Instagram performance reports by extracting metrics from screenshots using layout-aware, anchor-based OCR (OpenCV + Tesseract).

## Features

- **Anchor-Based OCR**: Locates metric fields relative to always-present text anchors ("Overview", "Views", "Follows", etc.) instead of fixed coordinates. Supports both iOS & Android screenshots across Reels and Posts.
- **Privacy-First**: No LLMs or third-party AI APIs used — runs open-source Tesseract OCR.
- **Multilingual Support**: UI and OCR search patterns support English, Spanish, French, and Portuguese.
- **Slide Deck & PDF Generator**: Renders beautiful presentation slides directly in the browser and exports print-ready PDFs.

## Repo Structure

```
.
├── index.html          # Frontend web app (hosted on GitHub Pages)
├── render.yaml         # Render Blueprint for automated backend deployment
├── README.md           # Documentation
└── backend/            # Python Flask OCR Microservice
    ├── app.py          # Flask API server
    ├── ocr_parser.py   # Anchor-based OCR parsing logic
    ├── requirements.txt # Python package dependencies
    └── Dockerfile      # Docker build file with Tesseract dependencies
```

## Getting Started (Local Development)

### 1. Backend Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

*Note: Ensure Tesseract OCR and language data packages are installed on your local operating system (`brew install tesseract tesseract-lang` on macOS).*

### 2. Frontend Setup

Serve `index.html` with any local HTTP server:
```bash
python3 -m http.server 8000
```
Open `http://localhost:8000` in your browser. Enter `http://localhost:5000` in the **Backend URL** field to test local extraction.

## Deployment

### Frontend Deployment (GitHub Pages)

1. Push this repository to GitHub under `chatgptricks/ig-report-generator`.
2. Navigate to **Settings → Pages** in your GitHub repository.
3. Under **Build and deployment**, set the source to `Deploy from branch`, select branch `main` and folder `/ (root)`.
4. Your site will be published at `https://chatgptricks.github.io/ig-report-generator/`.

### Backend Deployment (Render)

1. Sign in to [Render](https://render.com).
2. Click **New +** → **Blueprint** (or **Web Service**).
3. Connect your GitHub repository `chatgptricks/ig-report-generator`.
4. Render will automatically detect `render.yaml` and configure the Docker Web Service environment.
5. Once deployed, copy your Render URL (e.g. `https://ig-report-ocr-backend.onrender.com`) and paste it into the **Backend URL** input field on the frontend.
