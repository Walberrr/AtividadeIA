from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import pickle

# Mesmo tamanho usado no treino
MAX_LEN = 150

app = Flask(__name__)

# ===========================
# 1. CARREGAR O MODELO
# ===========================
model = load_model("news_model.h5")

# ===========================
# 2. CARREGAR TOKENIZER
# ===========================
with open("tokenizer.json", "r") as f:
    tokenizer_data = json.load(f)
tokenizer = tokenizer_from_json(tokenizer_data)

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

    # Preprocessamento idêntico ao treino
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
    app.run(host="0.0.0.0", port=8000)
