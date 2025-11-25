from flask import Flask, request, jsonify
import numpy as np

# ===========================
# Configurações
# ===========================
MAX_LEN = 150
model = None
tokenizer = None
encoder = None

app = Flask(__name__)

# ===========================
# CARREGAR RECURSOS SOMENTE NA PRIMEIRA REQUISIÇÃO
# ===========================
@app.before_first_request
def load_resources():
    global model, tokenizer, encoder
    import json, pickle
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.text import tokenizer_from_json

    print("Carregando modelo, tokenizer e encoder...")
    model = load_model("news_model.h5")

    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

    with open("label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    print("Recursos carregados com sucesso!")

# ===========================
# ROTAS
# ===========================
@app.route("/", methods=["GET"])
def home():
    return "API PLN Online e Funcionando!"

@app.route("/prever", methods=["POST"])
def prever():
    global model, tokenizer, encoder

    # Verifica se recursos estão carregados
    if model is None or tokenizer is None or encoder is None:
        return jsonify({"erro": "Modelo ainda não carregado, tente novamente em alguns segundos."}), 503

    data = request.get_json()
    texto = data.get("texto", "")

    # Preprocessamento
    seq = tokenizer.texts_to_sequences([texto])
    seq_pad = np.array(seq)
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    seq_pad = pad_sequences(seq_pad, maxlen=MAX_LEN)

    # Previsão
    pred = model.predict(seq_pad)
    classe = np.argmax(pred, axis=1)[0]
    categoria = encoder.inverse_transform([classe])[0]

    return jsonify({
        "input": texto,
        "categoria_prevista": str(categoria)
    })

# ===========================
# EXECUÇÃO
# ===========================
if __name__ == "__main__":
    # Rodar localmente para teste
    app.run(host="0.0.0.0", port=8000)
