from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle, re, os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(title="LSTM Text Prediction API", version="1.0")

# CORS Middleware (Fix OPTIONS 405 Error)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SEQ_LENGTH = 5   # Must match your notebook's SEQ_LENGTH

# API Key setup
API_KEY        = "abe2a618b8cb7931bbd11c56a6da4e9c7d9f9fc6ec2402836ec9bcb0a28339fb"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

# --- Load model and tokenizer once at startup ---
print("Loading model...")
model = load_model("models/best_lstm_model.h5")

print("Loading tokenizer...")
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

index_to_word = {v: k for k, v in tokenizer.word_index.items()}
print("Ready ✓")

# Input models (what the API receives)
class PredictRequest(BaseModel):
    seed_text: str
    n_words: int = 5
    temperature: float = 0.5

class TopKRequest(BaseModel):
    seed_text: str
    k: int = 5

# Output models (what the API returns)
class PredictResponse(BaseModel):
    seed_text: str
    generated_text: str
    full_output: str

class WordProbability(BaseModel):
    word: str
    probability: float

class TopKResponse(BaseModel):
    seed_text: str
    predictions: list[WordProbability]

def clean(text):
    return re.sub(r"[^a-z\s]", "", text.lower()).strip()

def predict_next_words(seed_text, n_words=5, temperature=0.5):
    result = seed_text
    current = clean(seed_text)
    for _ in range(n_words):
        tokens = tokenizer.texts_to_sequences([current])[0]
        tokens = pad_sequences([tokens], maxlen=SEQ_LENGTH, padding="pre")
        probs  = model.predict(tokens, verbose=0)[0]

        probs  = np.log(probs + 1e-8) / temperature
        probs  = np.exp(probs) / np.sum(np.exp(probs))

        idx    = np.random.choice(len(probs), p=probs)
        word   = index_to_word.get(idx, "")

        if word:
            result += " " + word
            current += " " + word

    return result

def get_top_k(seed_text, k=5):
    tokens = tokenizer.texts_to_sequences([clean(seed_text)])[0]
    tokens = pad_sequences([tokens], maxlen=SEQ_LENGTH, padding="pre")
    probs  = model.predict(tokens, verbose=0)[0]
    top_k  = np.argsort(probs)[-k:][::-1]

    return [{"word": index_to_word.get(int(i), "<UNK>"),
             "probability": round(float(probs[i]), 4)} for i in top_k]

@app.get("/")
def root():
    return {"status": "running", "model": "LSTM Text Prediction"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _=Security(verify_key)):
    try:
        full = predict_next_words(req.seed_text, req.n_words, req.temperature)
        generated = full[len(req.seed_text):].strip()

        return {
            "seed_text": req.seed_text,
            "generated_text": generated,
            "full_output": full
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/top-k")
def top_k(req: TopKRequest, _=Security(verify_key)):
    try:
        predictions = get_top_k(req.seed_text, req.k)
        return {"seed_text": req.seed_text, "predictions": predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}