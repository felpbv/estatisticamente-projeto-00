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
import re

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================================================
# 🔥 CONFIG
# ================================================
EMBEDDING_MODEL = "text-embedding-3-large"
ARQUIVO_EMBEDDINGS = f"embeddings_{EMBEDDING_MODEL}.pkl"

# ================================================
# 🚀 FLASK
# ================================================
app = Flask(__name__)
CORS(app)

# ================================================
# 📚 CARREGAR CSV
# ================================================
def carregar_base_csv(caminho):
    return pd.read_csv(caminho)

# ================================================
# 📄 CARREGAR PDF
# ================================================
def carregar_base_pdf(caminho_pdf):

    reader = PdfReader(caminho_pdf)
    textos = []

    for page in reader.pages:
        texto = page.extract_text()
        if texto:
            chunks = [texto[i:i+500] for i in range(0, len(texto), 500)]
            textos.extend(chunks)

    return pd.DataFrame({
        "categoria": "cardapio",
        "informacao": textos
    })

# ================================================
# ✂️ CHUNKING
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
# 🧠 EMBEDDINGS
# ================================================
def gerar_embeddings(df):

    textos = df["informacao"].tolist()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=textos
    )

    df["embedding"] = [e.embedding for e in response.data]

    return df

def salvar_embeddings(df):
    with open(ARQUIVO_EMBEDDINGS, "wb") as f:
        pickle.dump(df, f)

def carregar_embeddings():
    with open(ARQUIVO_EMBEDDINGS, "rb") as f:
        return pickle.load(f)

# ================================================
# 📐 SIMILARIDADE
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
        model=EMBEDDING_MODEL,
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

# ================================================
# 🎨 FORMATAÇÃO
# ================================================
def formatar_resposta(texto):
    texto = texto.replace("•", "<br>•")
    texto = texto.replace("- ", "<br>• ")
    return texto

def transformar_links(texto):
    url_pattern = r'(https?://[^\s]+)'

    def substituir(match):
        url = match.group(0)

        if "cardapio" in url.lower():
            texto_link = "📖 Ver cardápio"
        elif "reserva" in url.lower():
            texto_link = "📅 Fazer reserva"
        else:
            texto_link = "🔗 Acessar link"

        return f'<a href="{url}" target="_blank" style="color:#c58b2a;font-weight:bold;">{texto_link}</a>'

    return re.sub(url_pattern, substituir, texto)

def transformar_links_markdown(texto):
    pattern = r'\[(.*?)\]\((https?://[^\s]+)\)'

    def substituir(match):
        label = match.group(1)
        url = match.group(2)

        return f'<a href="{url}" target="_blank" style="color:#c58b2a;font-weight:bold;">{label}</a>'

    return re.sub(pattern, substituir, texto)

# ================================================
# 🤖 HISTÓRICO
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
Base:

{contexto}

Pergunta:
{pergunta}
"""
    })

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=historico
    )

    resposta = response.choices[0].message.content

    # 🔥 processamento correto
    resposta = transformar_links_markdown(resposta)
    resposta = transformar_links(resposta)
    resposta = formatar_resposta(resposta)

    historico.append({
        "role": "assistant",
        "content": resposta
    })

    return resposta

# ================================================
# 🚀 CARREGAR BASE
# ================================================
print("🔄 Inicializando base...")

# if os.path.exists(ARQUIVO_EMBEDDINGS):

#     print("⚡ Carregando embeddings...")
#     df_base = carregar_embeddings()

if False:

    # print("⚡ Carregando embeddings...")
    df_base = carregar_embeddings()

else:

    print("📚 CSV...")
    df_csv = carregar_base_csv("base-restaurante.csv")

    print("📄 PDF...")
    df_pdf = carregar_base_pdf("cardapio.pdf")

    print("🔗 Unindo...")
    df_base = pd.concat([df_csv, df_pdf], ignore_index=True)

    print("✂️ Chunking...")
    df_base = criar_chunks(df_base)

    print("🧠 Embeddings...")
    df_base = gerar_embeddings(df_base)

    print("💾 Salvando...")
    salvar_embeddings(df_base)

print("✅ Pronto!")

# ================================================
# 🌐 ROTAS
# ================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/perguntar")
def perguntar():
    try:
        pergunta = request.args.get("pergunta")

        if not pergunta:
            return jsonify({"erro": "Pergunta não enviada."})

        resposta = gerar_resposta(pergunta, df_base)

        return jsonify({"resposta": resposta})

    except Exception as e:
        print("ERRO:", str(e))
        return jsonify({"erro": str(e)})

# ================================================
# 🚀 RUN
# ================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
