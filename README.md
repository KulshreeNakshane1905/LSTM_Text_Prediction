# 🧠 LSTM-Based Text Prediction System
### Lab Assignment 5

---

## 📁 Project Structure

```
lstm_project/
│
├── LSTM_Text_Prediction.ipynb   ← Main notebook (model development)
├── requirements.txt             ← All Python dependencies
├── README.md                    ← This file
│
├── api/
│   └── app.py                   ← FastAPI server (called by n8n)
│
├── n8n/
│   └── lstm_workflow.json       ← n8n workflow export (import this)
│
└── model/                       ← Created after running notebook
    ├── lstm_text_prediction_model.h5
    ├── tokenizer.pkl
    ├── model_config.pkl
    └── training_curves.png
```

---

## 🚀 Step-by-Step: Run in VS Code

### STEP 1 — Prerequisites

Install these first:
- [VS Code](https://code.visualstudio.com/)
- Python 3.10 (recommended) — [python.org](https://python.org)
- [n8n Desktop](https://n8n.io/get-started/) OR n8n Cloud (free tier)

---

### STEP 2 — Open Project in VS Code

```bash
# Open the lstm_project/ folder in VS Code
# Then open a terminal: Ctrl + ` (backtick)
```

---

### STEP 3 — Create Virtual Environment

```bash
# In VS Code terminal:
python -m venv venv

# Activate it:
# Windows:
venv\Scripts\activate

# Mac / Linux:
source venv/bin/activate
```

---

### STEP 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

### STEP 5 — Run the Notebook (Train the Model)

1. Open `LSTM_Text_Prediction.ipynb` in VS Code
2. Top-right → **Select Kernel** → choose `venv` Python
3. Install the **Jupyter** extension if prompted
4. Run cells **one by one** (Shift+Enter):
   - Cell 1: Install deps
   - Cell 2: Imports
   - Cell 3: Download Wikipedia data
   - Cell 4: Preprocess text
   - Cell 5: Tokenize + generate sequences
   - Cell 6: Prepare X, y
   - Cell 7: Build LSTM model
   - Cell 8: **Train** (this takes 5–20 min depending on your machine)
   - Cell 9: Plot training curves
   - Cell 10: Define prediction functions
   - Cell 11: Test predictions (your demo output)
   - Cell 12: **Save model files** → creates `model/` folder

> ⚠️ After Cell 12 you will see:
> ```
> model/lstm_text_prediction_model.h5
> model/tokenizer.pkl
> model/model_config.pkl
> ```

---

### STEP 6 — Start the FastAPI Server

Open a **new terminal** in VS Code (keep notebook terminal open):

```bash
# Make sure venv is active
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# Start FastAPI server
cd api
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
 Model loaded  | vocab=XXXX | seq_len=10
```

**Test it in browser:** http://localhost:8000/docs
- Click `/predict` → Try it out → Enter your seed_text → Execute

---

### STEP 7 — Set Up n8n Workflow

#### Option A: n8n Desktop (easiest)

1. Download & install n8n Desktop from https://n8n.io/get-started/
2. Open n8n → **New Workflow**
3. Click **...** (top right) → **Import from File**
4. Select `n8n/lstm_workflow.json`
5. Click **Activate** (toggle top right)
6. Your webhook URL will be: `http://localhost:5678/webhook/lstm-predict`

#### Option B: n8n Cloud

1. Sign up at https://app.n8n.cloud (free tier available)
2. Import `lstm_workflow.json`
3. Change the HTTP Request node URL from `http://localhost:8000/predict`
   to your **ngrok tunnel URL** (see below)

#### Exposing local FastAPI to n8n Cloud (if needed):

```bash
# Install ngrok: https://ngrok.com/download
ngrok http 8000

# Copy the https://xxxx.ngrok.io URL
# Paste it in n8n HTTP Request node URL field
```

---

### STEP 8 — Test the n8n Workflow

#### Using Postman or curl:

```bash
# Test n8n webhook (n8n Desktop):
curl -X POST http://localhost:5678/webhook/lstm-predict \
  -H "Content-Type: application/json" \
  -d '{"seed_text": "artificial intelligence is used in", "n_words": 5}'
```

#### Expected Response:
```json
{
  "status": "success",
  "input": "artificial intelligence is used in",
  "predicted_text": "artificial intelligence is used in many fields of",
  "top_5_words": [
    {"word": "many", "probability": 0.2341},
    {"word": "various", "probability": 0.1823},
    ...
  ],
  "model_info": {
    "vocab_size": 4521,
    "seq_length": 10,
    "n_words": 5,
    "temperature": 0.8
  },
  "timestamp": "2026-04-21T10:30:00.000Z"
}
```

---

## n8n Workflow Architecture

```
[User / Postman]
      │  POST { seed_text, n_words }
      ▼
[n8n Webhook Node]           ← Entry point
      │
      ▼
[HTTP Request Node]          ← Calls FastAPI at localhost:8000/predict
      │
      ▼
[FastAPI Server (app.py)]    ← Loads LSTM model → runs prediction
      │
      ▼
[Code Node]                  ← Formats JSON response
      │
      ▼
[Respond to Webhook]         ← Returns prediction to caller
```

---
## 👥 Group Members

| Name | Contribution |
|------|-------------|
| Member 1 Preeti Koli | Dataset collection, preprocessing (Cells 3–6) |
| Member 2 Sakshi Bhingarkar | LSTM model design & training (Cells 7–9) |
| Member 3 Vaishnavi Thorave | Prediction functions, testing (Cells 10–11) |
| Member 4 Kulshree Nakshane | FastAPI server + n8n deployment |

---

