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

# 🔐 Usa variável de ambiente (CORRETO)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================================================
# 🔥 CONFIG
# ================================================
EMBEDDING_MODEL = "text-embedding-3-small"
ARQUIVO_EMBEDDINGS = "embeddings.pkl"

# ================================================
# 🚀 FLASK
# ================================================
app = Flask(__name__)
CORS(app)

# ================================================
# 📚 CARREGAMENTO DAS BASES
# ================================================
def carregar_base_csv(caminho):
    return pd.read_csv(caminho)

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
# 🔎 BUSCA CONTEXTO (RAG)
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

    return [
        f"{row['categoria']}: {row['informacao']}"
        for _, row in top.iterrows()
    ]

# ================================================
# 🎨 FORMATAÇÃO DE RESPOSTA
# ================================================
def transformar_links(texto):
    """
    Converte:
    [Texto](link)
    em botão HTML clicável (mobile-friendly)
    """
    pattern = r'\[(.*?)\]\((https?://[^\s)]+)\)'

    def substituir(match):
        label = match.group(1)
        url = match.group(2)
        return f'<a href="{url}" target="_blank" class="link-btn">{label}</a>'

    return re.sub(pattern, substituir, texto)

def formatar_lista(texto):
    linhas = texto.split("\n")
    linhas = [l.strip() for l in linhas if l.strip()]

    if not any("•" in l for l in linhas):
        linhas = [f"• {l}" for l in linhas]

    return "<br>".join(linhas)

def formatar_resposta(texto):
    texto = transformar_links(texto)
    texto = formatar_lista(texto)
    return texto

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

Regras:
- Seja educado
- Fale levemente como mineiro
- Seja direto
- Responda em lista
- Use frases curtas
- Quando tiver link, use:

[Ver Cardápio](URL)
"""
    }
]

# ================================================
# 🤖 GERA RESPOSTA
# ================================================
def gerar_resposta(pergunta, df):

    contexto = "\n".join(buscar_contexto(pergunta, df))

    historico.append({
        "role": "user",
        "content": f"Base:\n{contexto}\nPergunta:\n{pergunta}"
    })

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=historico
    )

    resposta = response.choices[0].message.content

    resposta = formatar_resposta(resposta)

    historico.append({
        "role": "assistant",
        "content": resposta
    })

    return resposta

# ================================================
# 🚀 INICIALIZAÇÃO
# ================================================
print("🔄 Inicializando...")

if os.path.exists(ARQUIVO_EMBEDDINGS):
    print("⚡ Carregando embeddings salvos...")
    df_base = carregar_embeddings()
else:
    print("📚 Criando nova base...")

    df_csv = carregar_base_csv("base-restaurante.csv")
    df_pdf = carregar_base_pdf("cardapio.pdf")

    df_base = pd.concat([df_csv, df_pdf], ignore_index=True)
    df_base = criar_chunks(df_base)
    df_base = gerar_embeddings(df_base)

    salvar_embeddings(df_base)

print("✅ Sistema pronto!")

# ================================================
# 🌐 ROTAS
# ================================================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/perguntar")
def perguntar():
    pergunta = request.args.get("pergunta")

    if not pergunta:
        return jsonify({"erro": "Pergunta não enviada"})

    resposta = gerar_resposta(pergunta, df_base)
    return jsonify({"resposta": resposta})

# ================================================
# 🚀 RUN
# ================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
