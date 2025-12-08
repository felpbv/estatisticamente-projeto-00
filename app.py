# ================================================
# 🚀 IMPORTS NECESSÁRIOS
# ================================================
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from openai import OpenAI
import os

# Inicializa cliente OpenAI
client = OpenAI(api_key=os.getenv("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"))

# ================================================
# 🚀 INICIALIZAÇÃO DO FLASK + CORS
# ================================================
app = Flask(__name__)
CORS(app)  # Permite chamadas do frontend

# ================================================
# 1) Carrega a base de conhecimento local (CSV)
# ================================================
def carregar_base_csv(caminho_csv):
    df = pd.read_csv(caminho_csv)
    return df


# ================================================
# 2) Gera embeddings dos textos da base
# ================================================
def gerar_embeddings_base(df):
    textos = df["informacao"].tolist()

    embeddings_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=textos
    )

    df["embedding"] = [e.embedding for e in embeddings_response.data]
    return df


# ================================================
# 3) Similaridade de cosseno
# ================================================
def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# ================================================
# 4) Busca os trechos mais relevantes
# ================================================
def buscar_contexto(pergunta, df, top_k=3):
    pergunta_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=pergunta
    ).data[0].embedding

    df["similaridade"] = df["embedding"].apply(
        lambda e: cosine_similarity(e, pergunta_emb)
    )

    top_resultados = df.sort_values("similaridade", ascending=False).head(top_k)

    contexto = "\n".join(
        f"{row['categoria']}: {row['informacao']}"
        for _, row in top_resultados.iterrows()
    )
    return contexto


# ================================================
# 5) Gera a resposta do modelo com RAG
# ================================================
historico = [
    {"role": "system", "content": "Você é um assistente especializado no restaurante Sabor da Serra."}
]

def gerar_resposta(pergunta, df):
    contexto_relevante = buscar_contexto(pergunta, df)

    historico.append({
        "role": "user",
        "content": f"Base de conhecimento:\n{contexto_relevante}\n\nPergunta: {pergunta}"
    })

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=historico
    )

    resposta = response.choices[0].message.content
    historico.append({"role": "assistant", "content": resposta})
    return resposta


# ================================================
# 🔥 Carregamos a base AO INICIAR A APLICAÇÃO
# Funciona no Render e localmente
# ================================================
print("🔄 Carregando base e gerando embeddings...")
df_base = carregar_base_csv("base-restaurante.csv")
df_base = gerar_embeddings_base(df_base)
print("✅ Base pronta!")


# ================================================
# 🚀 ROTA 1 — Frontend
# ================================================
@app.route("/")
def index():
    return render_template("index.html")


# ================================================
# 🚀 ROTA 2 — API
# ================================================
@app.route("/perguntar")
def perguntar():
    pergunta = request.args.get("pergunta")

    if not pergunta:
        return jsonify({"erro": "Pergunta não enviada."})

    resposta = gerar_resposta(pergunta, df_base)
    return jsonify({"resposta": resposta})


# ================================================
# 🚀 INICIALIZAÇÃO CORRETA PARA O RENDER
# ================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

