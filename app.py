"""
LSTM Text Prediction — FastAPI Server
======================================
This server exposes the trained LSTM model via a REST API.
n8n calls the /predict endpoint via its HTTP Request node.

Run:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import re
import pickle
import numpy as np
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="LSTM Text Prediction API",
    description=(
        "Predict the next word(s) in a sequence using a trained LSTM model. "
        "Called by n8n AI Agent workflow."
    ),
    version="1.0.0",
)

# Allow all origins (needed for n8n cloud → local tunnel, or testing from browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  Load model artifacts at startup
# ─────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent           # LSTM/
MODEL_DIR = BASE_DIR / "models"             # LSTM/models/

try:
    # Use best_lstm_model.h5 (saved by ModelCheckpoint during training)
    model     = load_model(str(MODEL_DIR / "best_lstm_model.h5"))

    # tokenizer.pkl is inside LSTM/models/
    with open(MODEL_DIR / "tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    # model_config.pkl is in LSTM/ root (not inside models/)
    with open(BASE_DIR / "model_config.pkl", "rb") as f:
        config    = pickle.load(f)

    SEQ_LENGTH = config["seq_length"]
    VOCAB_SIZE = config["vocab_size"]
    print(f" Model loaded  | vocab={VOCAB_SIZE} | seq_len={SEQ_LENGTH}")

except FileNotFoundError as e:
    raise RuntimeError(
        f"Model files not found: {e}\n"
        "Make sure the following files exist:\n"
        "  LSTM/models/best_lstm_model.h5\n"
        "  LSTM/models/tokenizer.pkl\n"
        "  LSTM/model_config.pkl\n"
        "Run the notebook first (all cells) to generate these files."
    )


# ─────────────────────────────────────────────
#  Request / Response schemas
# ─────────────────────────────────────────────
class PredictRequest(BaseModel):
    seed_text   : str
    n_words     : int   = 5      # how many words to predict
    top_k       : int   = 5      # how many top candidates to return
    temperature : float = 0.8    # sampling temperature


class WordProbability(BaseModel):
    word        : str
    probability : float


class PredictResponse(BaseModel):
    seed_text    : str
    predicted    : str
    top_k_words  : list[WordProbability]
    model_info   : dict


# ─────────────────────────────────────────────
#  Helper functions
# ─────────────────────────────────────────────
def clean(text: str) -> str:
    """Normalize input text the same way training did."""
    return re.sub(r"[^a-z\s]", "", text.lower()).strip()


def predict_continuation(seed_text: str, n_words: int, temperature: float) -> str:
    result  = seed_text
    current = clean(seed_text)

    for _ in range(n_words):
        tokens = tokenizer.texts_to_sequences([current])[0]
        tokens = pad_sequences([tokens], maxlen=SEQ_LENGTH, padding="pre")
        preds  = model.predict(tokens, verbose=0)[0]

        # Temperature scaling
        preds = np.log(preds + 1e-8) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))

        idx   = np.random.choice(len(preds), p=preds)
        word  = next(
            (w for w, i in tokenizer.word_index.items() if i == idx), ""
        )
        if word:
            result  += " " + word
            current += " " + word

    return result


def top_k_predictions(seed_text: str, k: int) -> list[dict]:
    current = clean(seed_text)
    tokens  = tokenizer.texts_to_sequences([current])[0]
    tokens  = pad_sequences([tokens], maxlen=SEQ_LENGTH, padding="pre")
    preds   = model.predict(tokens, verbose=0)[0]

    top_indices = np.argsort(preds)[-k:][::-1]
    idx_to_word = {v: kk for kk, v in tokenizer.word_index.items()}

    return [
        {"word": idx_to_word.get(int(i), "<UNK>"), "probability": round(float(preds[i]), 4)}
        for i in top_indices
    ]


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    """Health check — used by n8n to verify server is running."""
    return {
        "status"    : "running",
        "model"     : "LSTM Text Prediction",
        "endpoints" : ["/predict", "/docs"]
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Predict next words given a seed text.

    **n8n calls this endpoint** via its HTTP Request node:
    - Method : POST
    - URL    : http://localhost:8000/predict
    - Body   : { "seed_text": "...", "n_words": 5 }
    """
    if not req.seed_text.strip():
        raise HTTPException(status_code=400, detail="seed_text cannot be empty.")
    if req.n_words < 1 or req.n_words > 50:
        raise HTTPException(status_code=400, detail="n_words must be between 1 and 50.")
    if req.temperature <= 0:
        raise HTTPException(status_code=400, detail="temperature must be > 0.")

    continuation = predict_continuation(req.seed_text, req.n_words, req.temperature)
    top_k        = top_k_predictions(req.seed_text, req.top_k)

    return PredictResponse(
        seed_text   = req.seed_text,
        predicted   = continuation,
        top_k_words = top_k,
        model_info  = {
            "vocab_size" : VOCAB_SIZE,
            "seq_length" : SEQ_LENGTH,
            "n_words"    : req.n_words,
            "temperature": req.temperature,
        }
    )