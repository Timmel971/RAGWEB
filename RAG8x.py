import os
import pandas as pd
import PyPDF2
import openai
import gdown
import tempfile
import requests
import numpy as np
import logging
from io import BytesIO
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable
from openai import OpenAI
from typing import List, Tuple, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import hashlib
import pickle
import time

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API-Key und Neo4j-Konfiguration (hier aus Umgebungsvariablen oder Secrets laden)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GDRIVE_URL = os.getenv("GDRIVE_URL")

if not all([OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GDRIVE_URL]):
    logger.error("❌ Umgebungsvariablen fehlen")
    raise ValueError("Bitte setze OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD und GDRIVE_URL")

client = openai.OpenAI(api_key=OPENAI_API_KEY)
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# FastAPI-App
app = FastAPI()

# CORS für CodePen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Für Produktion spezifische Origins wie "https://codepen.io" setzen
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Globale Variablen
DOWNLOAD_PATH = tempfile.mkdtemp()
EMBEDDING_CACHE_FILE = os.path.join(DOWNLOAD_PATH, "embedding_cache.pkl")
embedding_cache: Dict[str, np.ndarray] = {}
chunk_embeddings: List[Tuple[str, np.ndarray]] = []
documents: List[str] = []

# Funktionen aus deinem Code (unverändert)
def verify_neo4j_connection() -> bool:
    try:
        with driver.session() as session:
            session.run("MATCH (n) RETURN n LIMIT 1")
        logger.info("✅ Neo4j-Verbindung erfolgreich")
        return True
    except ServiceUnavailable as e:
        logger.error(f"❌ Neo4j-Verbindung fehlgeschlagen: {e}")
        return False

def load_embedding_cache() -> None:
    global embedding_cache
    if os.path.exists(EMBEDDING_CACHE_FILE):
        try:
            with open(EMBEDDING_CACHE_FILE, 'rb') as f:
                embedding_cache = pickle.load(f)
            logger.info("✅ Embedding-Cache geladen")
        except Exception as e:
            logger.warning(f"❌ Fehler beim Laden des Embedding-Cache: {e}")

def save_embedding_cache() -> None:
    try:
        with open(EMBEDDING_CACHE_FILE, 'wb') as f:
            pickle.dump(embedding_cache, f)
        logger.info("✅ Embedding-Cache gespeichert")
    except Exception as e:
        logger.warning(f"❌ Fehler beim Speichern des Embedding-Cache: {e}")

def download_drive_folder(output_path: str) -> None:
    try:
        folder_id = GDRIVE_URL.split('/')[-1]
        gdown.download_folder(id=folder_id, output=output_path, quiet=False)
        logger.info(f"📥 Google Drive-Ordner erfolgreich heruntergeladen nach {output_path}")
    except Exception as e:
        logger.error(f"❌ Fehler beim Herunterladen des Google Drive-Ordners: {e}")
        raise

def read_folder_data(folder_path: str) -> List[str]:
    files_data = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".pdf"):
            try:
                pdf_text = []
                with open(file_path, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            pdf_text.append(text)
                if pdf_text:
                    files_data.append(" ".join(pdf_text))
            except Exception as e:
                logger.error(f"❌ Fehler beim Lesen der PDF {file_name}: {e}")
    return files_data

def split_text(text: str, max_length: int = 300) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        if current_length + len(word) + 1 > max_length and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(word)
        current_length += len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def get_embedding(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash in embedding_cache:
        return embedding_cache[text_hash]
    try:
        response = client.embeddings.create(model=model, input=[text])
        embedding = np.array(response.data[0].embedding)
        embedding_cache[text_hash] = embedding
        save_embedding_cache()
        return embedding
    except openai.OpenAIError as e:
        logger.error(f"❌ OpenAI Embedding-Fehler: {e}")
        return np.zeros(1536)

def create_embeddings_parallel(documents: List[str], max_length: int = 300) -> List[Tuple[str, np.ndarray]]:
    chunk_embeddings = []
    chunks = [chunk for doc in documents for chunk in split_text(doc, max_length)]
    if not chunks:
        logger.warning("⚠️ Keine Chunks zum Verarbeiten")
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_chunk = {executor.submit(get_embedding, chunk): chunk for chunk in chunks}
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                embedding = future.result()
                if not np.all(embedding == 0):
                    chunk_embeddings.append((chunk, embedding))
            except Exception as e:
                logger.error(f"❌ Fehler bei Embedding für Chunk: {e}")
    return chunk_embeddings

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1

def retrieve_relevant_chunks(query: str, chunk_embeddings: List[Tuple[str, np.ndarray]], top_n: int = 2) -> str:
    try:
        query_emb = get_embedding(query)
        if np.all(query_emb == 0):
            logger.warning("⚠️ Ungültiges Query-Embedding")
            return "Keine relevanten Dokumente gefunden."
        similarities = [(chunk, cosine_similarity(query_emb, emb)) for chunk, emb in chunk_embeddings]
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [chunk for chunk, _ in similarities[:top_n] if chunk]
        max_tokens = 4000
        context = "\n\n".join(top_chunks)
        if estimate_tokens(context) > max_tokens:
            context = context[:max(1, int(len(context) * max_tokens / estimate_tokens(context)))]
            logger.warning("⚠️ Dokumenten-Kontext gekürzt")
        return context if context else "Keine relevanten Dokumente gefunden."
    except Exception as e:
        logger.error(f"❌ Fehler bei der Dokumentensuche: {e}")
        return "Fehler bei der Dokumentensuche."

def get_neo4j_context(user_query: str, limit: int = 100) -> str:
    with driver.session() as session:
        try:
            context_lines = []
            max_tokens = 8000
            result_metrics = session.run("""
                MATCH (m:FinancialMetric)-[r1]-(n)
                WHERE m.value IS NOT NULL AND m.year IS NOT NULL
                RETURN m, r1, n
                LIMIT $limit
            """, limit=limit//2)
            query_lower = user_query.lower()
            company_filter = ""
            if "siemens" in query_lower or "tass" in query_lower:
                company_filter = "WHERE p.name CONTAINS 'Siemens' OR c.name CONTAINS 'TASS'"
            result_companies = session.run(f"""
                MATCH (p:ParentCompany)-[r2:HAS_PARTICIPATION|HAS_ACQUISITION]->(c:Company)
                {company_filter}
                RETURN p, r2, c
                LIMIT $limit
            """, limit=limit//2)

            def describe_node(node):
                if node is None:
                    return "None"
                label = list(node.labels)[0] if node.labels else "Node"
                name = node.get("name") or node.get("id") or label
                props = ", ".join([f"{k}: {v}" for k, v in node.items() if k not in ["name", "id"] and v is not None])
                return f"{label}({name}) [{props}]" if props else f"{label}({name})"

            for record in result_metrics:
                m = record["m"]
                n = record["n"]
                r1 = record["r1"]
                m_desc = describe_node(m)
                n_desc = describe_node(n)
                line = f"{m_desc} -[:{r1.type}]- {n_desc}"
                context_lines.append(line)

            for record in result_companies:
                p = record["p"]
                c = record["c"]
                r2 = record["r2"]
                p_desc = describe_node(p)
                c_desc = describe_node(c)
                props = ", ".join([f"{k}: {v}" for k, v in r2.items() if v is not None])
                line = f"{p_desc} -[:{r2.type} {props}]- {c_desc}"
                context_lines.append(line)

            context = "\n".join(context_lines)
            if estimate_tokens(context) > max_tokens:
                context_lines = context_lines[:max(1, int(len(context_lines) * max_tokens / estimate_tokens(context)))]
                context = "\n".join(context_lines)
                logger.warning("⚠️ Neo4j-Kontext gekürzt")
            return context or "Keine relevanten Daten in Neo4j gefunden."
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden der Neo4j-Daten: {e}")
            return "Fehler beim Laden der Neo4j-Daten."

def generate_response(context: str, user_query: str) -> str:
    if not context.strip():
        return "Keine ausreichenden Daten gefunden."
    max_tokens = 12000
    if estimate_tokens(context) > max_tokens:
        context = context[:max(1, int(len(context) * max_tokens / estimate_tokens(context)))]
        logger.warning("⚠️ Gesamter Kontext gekürzt")
    messages = [
        {
            "role": "system",
            "content": (
                "Du bist ein präziser Assistent für Siemens-Konzern-Analysen. "
                "Antworte ausschließlich auf Basis der bereitgestellten Daten. "
                "Nutze keine externen Quellen oder Vorwissen. "
                "Wenn die Frage nach Anteilsverhältnissen oder Übernahmen fragt, suche nach Beziehungen wie HAS_PARTICIPATION oder HAS_ACQUISITION und gib den sharePercentage an. "
                "Falls keine passenden Informationen im Kontext vorhanden sind, antworte mit 'Keine ausreichenden Daten gefunden'. "
                "Antworte klar, präzise und in deutscher Sprache."
            )
        },
        {"role": "user", "content": f"Kontext: {context}\nBenutzerfrage: {user_query}"}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=700,
            temperature=0.2,
            top_p=0.9
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        logger.error(f"❌ OpenAI API-Fehler: {e}")
        return "Fehler bei der Antwortgenerierung."

# Eingabemodell für API
class Query(BaseModel):
    question: str

# Initialisierung beim Start der API
@app.on_event("startup")
async def startup_event():
    global documents, chunk_embeddings
    if not verify_neo4j_connection():
        raise HTTPException(status_code=500, detail="Neo4j-Verbindung fehlgeschlagen")
    load_embedding_cache()
    download_drive_folder(DOWNLOAD_PATH)
    documents = read_folder_data(DOWNLOAD_PATH)
    if not documents:
        logger.warning("⚠️ Keine gültigen PDF-Dokumente gefunden")
        raise HTTPException(status_code=500, detail="Keine gültigen PDF-Dokumente gefunden")
    chunk_embeddings = create_embeddings_parallel(documents, max_length=300)
    if not chunk_embeddings:
        logger.warning("⚠️ Keine Embeddings erstellt")
        raise HTTPException(status_code=500, detail="Keine Embeddings erstellt")

# API-Endpunkt
@app.post("/analyze")
async def analyze(query: Query):
    try:
        graph_context = get_neo4j_context(query.question, limit=100)
        document_context = retrieve_relevant_chunks(query.question, chunk_embeddings, top_n=2)
        combined_context = f"Neo4j-Daten:\n{graph_context}\n\nGeschäftsberichte:\n{document_context}"
        answer = generate_response(combined_context, query.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"❌ Fehler bei der Verarbeitung: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
