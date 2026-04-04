# ================================================
# 🚀 IMPORTS
# ================================================
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from openai import OpenAI
from pypdf import PdfReader
import pickle
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================================================
# 🚀 FLASK
# ================================================
app = Flask(__name__)
CORS(app)

# ================================================
# 📚 CARREGAR CSV
# ================================================
def carregar_base_csv(caminho):

    df = pd.read_csv(caminho)

    return df


# ================================================
# 📄 CARREGAR PDF
# ================================================
def carregar_base_pdf(caminho_pdf):

    reader = PdfReader(caminho_pdf)

    textos = []

    for page in reader.pages:

        texto = page.extract_text()

        if texto:

            # divide texto em pedaços menores
            chunks = [texto[i:i+500] for i in range(0, len(texto), 500)]

            textos.extend(chunks)

    df_pdf = pd.DataFrame({
        "categoria": "cardapio",
        "informacao": textos
    })

    return df_pdf


# ================================================
# ✂️ CHUNKING (para textos grandes)
# ================================================
def criar_chunks(df):

    novos = []

    for _, row in df.iterrows():

        texto = row["informacao"]

        chunks = [texto[i:i+400] for i in range(0, len(texto), 400)]

        for chunk in chunks:
            novos.append({
                "categoria": row["categoria"],
                "informacao": chunk
            })

    return pd.DataFrame(novos)


# ================================================
# 🧠 GERAR EMBEDDINGS
# ================================================
def gerar_embeddings(df):

    textos = df["informacao"].tolist()

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=textos
    )

    df["embedding"] = [e.embedding for e in response.data]

    return df


# ================================================
# 💾 SALVAR EMBEDDINGS
# ================================================
def salvar_embeddings(df):

    with open("embeddings.pkl", "wb") as f:
        pickle.dump(df, f)


# ================================================
# 📂 CARREGAR EMBEDDINGS
# ================================================
def carregar_embeddings():

    with open("embeddings.pkl", "rb") as f:
        return pickle.load(f)


# ================================================
# 📐 COSINE SIMILARITY
# ================================================
def cosine_similarity(v1, v2):

    v1 = np.array(v1)
    v2 = np.array(v2)

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# ================================================
# 🔎 BUSCAR CONTEXTO
# ================================================
def buscar_contexto(pergunta, df, top_k=4):

    pergunta_emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=pergunta
    ).data[0].embedding

    df["similaridade"] = df["embedding"].apply(
        lambda e: cosine_similarity(e, pergunta_emb)
    )

    top = df.sort_values("similaridade", ascending=False).head(top_k)

    contexto = "\n".join(
        f"{row['categoria']}: {row['informacao']}"
        for _, row in top.iterrows()
    )

    return contexto

import re

# ================================================
# 📂 qUEBRA DE LINHAS
# ================================================

def formatar_resposta(texto):
    texto = texto.replace("•", "<br>•")
    texto = texto.replace("- ", "<br>• ")
    return texto
# ================================================
# 📂 TRANSFORMA LINKS
# ================================================
def transformar_links(texto):
    url_pattern = r'(https?://[^\s]+)'

    def substituir(match):
        url = match.group(0)

        # define texto baseado no conteúdo
        if "cardapio" in url.lower():
            texto_link = "📖 Ver cardápio"
        elif "reserva" in url.lower():
            texto_link = "📅 Fazer reserva"
        else:
            texto_link = "🔗 Acessar link"

        return f'<a href="{url}" target="_blank" style="color:#c58b2a;font-weight:bold;">{texto_link}</a>'

    return re.sub(url_pattern, substituir, texto)

# ================================================
# 🤖 HISTÓRICO DO CHAT
# ================================================
historico = [
    {
        "role": "system",
        "content": """
Você é um assistente especializado no restaurante Fogão Mineiro.
Seu nome é Mineirinho.

Responda de forma educada e clara com um leve sotaque mineiro.
Quando houver links, envie o hiperlink para o cliente clicar.
Responda de maneira simples, curta e em tópicos, para que a resposta seja simples e rápida.

FORMATAÇÃO OBRIGATÓRIA:
- Sempre responda em lista com quebra de linha
- Use um item por linha
- Use este formato:

• Item 1  
• Item 2  
• Item 3  

Nunca escreva tudo em uma única linha.
"""
    }
]


# ================================================
# 🤖 GERAR RESPOSTA
# ================================================
def gerar_resposta(pergunta, df):

    contexto = buscar_contexto(pergunta, df)

    historico.append({
        "role": "user",
        "content": f"""
Base de conhecimento:

{contexto}

Pergunta do cliente:
{pergunta}
"""
    })

    response = client.chat.completions.create(

        model="gpt-4.1-mini",

        messages=historico

    )

    resposta = response.choices[0].message.content
    
    resposta = formatar_resposta(resposta)
    resposta = transformar_links(resposta)
    
    historico.append({
        "role": "assistant",
        "content": resposta
    })

    return resposta


# ================================================
# 🚀 CARREGAMENTO DA BASE
# ================================================
print("🔄 Inicializando base de conhecimento...")

if os.path.exists("embeddings.pkl"):

    print("⚡ Carregando embeddings já prontos...")

    df_base = carregar_embeddings()

else:

    print("📚 Lendo CSV...")
    df_csv = carregar_base_csv("base-restaurante.csv")

    print("📄 Lendo PDF do cardápio...")
    df_pdf = carregar_base_pdf("cardapio.pdf")

    print("🔗 Unindo bases...")
    df_base = pd.concat([df_csv, df_pdf], ignore_index=True)

    print("✂️ Criando chunks...")
    df_base = criar_chunks(df_base)

    print("🧠 Gerando embeddings...")
    df_base = gerar_embeddings(df_base)

    print("💾 Salvando embeddings...")
    salvar_embeddings(df_base)

print("✅ Base pronta!")


# ================================================
# 🌐 ROTA FRONTEND
# ================================================
@app.route("/")
def index():
    return render_template("index.html")


# ================================================
# 🔎 API PERGUNTAR
# ================================================
@app.route("/perguntar")
def perguntar():

    pergunta = request.args.get("pergunta")

    if not pergunta:
        return jsonify({"erro": "Pergunta não enviada."})

    resposta = gerar_resposta(pergunta, df_base)

    return jsonify({"resposta": resposta})


# ================================================
# 🚀 EXECUÇÃO
# ================================================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(host="0.0.0.0", port=port)
