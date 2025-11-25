from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import pickle

MAX_LEN = 150
app = Flask(__name__)

# Variáveis globais
model = None
tokenizer = None
encoder = None

# Função de lazy load
def lazy_load_resources():
    global model, tokenizer, encoder
    if model is None:
        model = load_model("news_model.h5")
    if tokenizer is None:
        with open("tokenizer.json", "r", encoding="utf-8") as f:
            tokenizer_json = f.read()
            tokenizer = tokenizer_from_json(tokenizer_json)
    if encoder is None:
        with open("label_encoder.pkl", "rb") as f:
            encoder = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return "API PLN Online e Funcionando!"

@app.route("/prever", methods=["POST"])
def prever():
    lazy_load_resources()  # carrega modelo/tokenizer/encoder só quando necessário

    data = request.get_json()
    texto = data.get("texto", "")

    # Preprocessamento
    seq = tokenizer.texts_to_sequences([texto])
    seq_pad = pad_sequences(seq, maxlen=MAX_LEN)

    # Previsão
    pred = model.predict(seq_pad)
    classe = np.argmax(pred, axis=1)[0]
    categoria = encoder.inverse_transform([classe])[0]

    return jsonify({
        "input": texto,
        "categoria_prevista": str(categoria)
    })

if __name__ == "__main__":
    # Configuração para Render: 1 worker + 1 thread pra economizar memória
    from waitress import serve  # mais leve que gunicorn em ambientes limitados
    serve(app, host="0.0.0.0", port=8000)
