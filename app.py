# ================================================
# 🚀 IMPORTS NECESSÁRIOS
# ================================================
# Framework web para servir frontend + rotas da API
from flask import Flask, request, jsonify, render_template

# Manipulação de dados
import pandas as pd
import numpy as np

# Cliente OpenAI para embeddings e chat
from openai import OpenAI

# Foi assumido que você adicionou a chave no ambiente ou no código
client = OpenAI(api_key="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")

# ================================================
# 🚀 INICIALIZAÇÃO DO FLASK
# ================================================
app = Flask(__name__)

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

    # Chama o modelo de embeddings da OpenAI
    embeddings_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=textos
    )

    # Salva cada embedding na coluna do dataframe
    df["embedding"] = [e.embedding for e in embeddings_response.data]
    return df


# ================================================
# 3) Similaridade de cosseno
# ================================================
def cosine_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# ================================================
# 4) Busca os trechos mais relevantes da base
# ================================================
def buscar_contexto(pergunta, df, top_k=3):
    # Embedding da pergunta
    pergunta_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=pergunta
    ).data[0].embedding

    # Calcula similaridade com cada embedding da base
    df["similaridade"] = df["embedding"].apply(lambda e: cosine_similarity(e, pergunta_emb))

    # Seleciona os trechos mais relevantes
    top_resultados = df.sort_values("similaridade", ascending=False).head(top_k)

    # Junta os trechos selecionados
    contexto = "\n".join(f"{row['categoria']}: {row['informacao']}" 
                         for _, row in top_resultados.iterrows())
    return contexto


# ================================================
# 5) Gera resposta do modelo com o contexto RAG
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
# 🚀 ROTA 1: Página inicial (frontend)
# ================================================
@app.route("/")
def index():
    # Flask automaticamente busca templates/index.html
    return render_template("index.html")


# ================================================
# 🚀 ROTA 2: Endpoint da API para perguntas
# ================================================
@app.route("/perguntar")
def perguntar():
    pergunta = request.args.get("pergunta")

    if not pergunta:
        return jsonify({"erro": "Pergunta não enviada."})

    resposta = gerar_resposta(pergunta, df_base)
    return jsonify({"resposta": resposta})


# ================================================
# 🚀 INICIALIZAÇÃO DO SERVIDOR
# ================================================
if __name__ == "__main__":
    print("🔄 Carregando base e gerando embeddings...")
    df_base = carregar_base_csv("base-restaurante.csv")
    df_base = gerar_embeddings_base(df_base)
    print("✅ Pronto! Acesse http://127.0.0.1:5000")

    # Roda o flask
    app.run(debug=True)
