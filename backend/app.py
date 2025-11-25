from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import pickle
import os

# Mesmo tamanho usado no treino
MAX_LEN = 150

app = Flask(__name__)
CORS(app)  # <--- AQUI, MONA! LIBERA O FRONT PRA CHAMAR A API

# ===========================
# 1. CARREGAR O MODELO
# ===========================
model = load_model("news_model.h5")

# ===========================
# 2. CARREGAR TOKENIZER
# ===========================
with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_json = f.read()  # lê como string
tokenizer = tokenizer_from_json(tokenizer_json)

# ===========================
# 3. CARREGAR LABEL ENCODER
# ===========================
with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return "API PLN Online e Funcionando!"

@app.route("/prever", methods=["POST"])
def prever():
    data = request.get_json()
    texto = data.get("texto", "")

    seq = tokenizer.texts_to_sequences([texto])
    seq_pad = pad_sequences(seq, maxlen=MAX_LEN)

    pred = model.predict(seq_pad)
    classe = np.argmax(pred, axis=1)[0]

    categoria = encoder.inverse_transform([classe])[0]

    return jsonify({
        "input": texto,
        "categoria_prevista": str(categoria)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
