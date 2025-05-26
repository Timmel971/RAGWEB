import os
import pandas as pd
import PyPDF2
import openai
import gdown
import tempfile
import numpy as np
import logging
import json
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
import concurrent.futures
import threading

# Logging konfigurieren
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API-Key und Neo4j-Konfiguration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GDRIVE_URL = os.getenv("GDRIVE_URL")

if not all([OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, GDRIVE_URL]):
    logger.error("❌ Umgebungsvariablen fehlen")
    raise ValueError("Bitte setze OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD und GDRIVE_URL")

# Überprüfe das URI-Schema
valid_schemes = ['neo4j', 'neo4j+s', 'neo4j+ssc', 'bolt', 'bolt+s', 'bolt+ssc']
uri_scheme = NEO4J_URI.split('://')[0]
if uri_scheme not in valid_schemes:
    logger.error(f"❌ Ungültiges URI-Schema: {uri_scheme}. Erlaubte Schemata: {valid_schemes}")
    raise ValueError(f"Ungültiges URI-Schema: {uri_scheme}. Erlaubte Schemata: {valid_schemes}")

# Neo4j-Treiber initialisieren
try:
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD)
    )
except Exception as e:
    logger.error(f"❌ Fehler beim Initialisieren des Neo4j-Treibers: {e}")
    raise

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# FastAPI-App
app = FastAPI()

# CORS für CodePen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
cache_lock = threading.Lock()

# Funktionen
def verify_neo4j_connection() -> bool:
    try:
        with driver.session() as session:
            session.run("MATCH (n) RETURN n LIMIT 1")
        logger.info("✅ Neo4j-Verbindung erfolgreich")
        return True
    except ServiceUnavailable as e:
        logger.error(f"❌ Neo4j-Verbindungsfehlgeschlagen: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unerwarteter Fehler bei der Neo4j-Verbindung: {e}")
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
    with cache_lock:
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

def read_folder_data(folder_path: str, password: str = "") -> List[str]:
    files_data = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if file_name.endswith(".pdf"):
            try:
                pdf_text = []
                with open(file_path, "rb") as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    if pdf_reader.is_encrypted:
                        logger.info(f"🔒 PDF {file_name} ist verschlüsselt")
                        try:
                            pdf_reader.decrypt(password)
                            logger.info(f"✅ PDF {file_name} erfolgreich mit Passwort entschlüsselt")
                        except Exception as e:
                            logger.error(f"❌ Fehler beim Entschlüsseln der PDF {file_name}: {str(e)}")
                            continue
                    else:
                        logger.info(f"✅ PDF {file_name} ist nicht verschlüsselt")
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            pdf_text.append(text)
                if pdf_text:
                    files_data.append(" ".join(pdf_text))
                    logger.info(f"📄 Text aus PDF {file_name} erfolgreich extrahiert")
                else:
                    logger.warning(f"⚠️ Kein Text in PDF {file_name} extrahiert")
            except Exception as e:
                logger.error(f"❌ Fehler beim Lesen der PDF {file_name}: {str(e)}")
                continue
    if not files_data:
        logger.warning("⚠️ Keine gültigen PDF-Dokumente gefunden")
    else:
        logger.info(f"🎉 {len(files_data)} PDF-Dokumente erfolgreich verarbeitet")
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
    with cache_lock:
        if text_hash in embedding_cache:
            return embedding_cache[text_hash]
    try:
        response = client.embeddings.create(model=model, input=[text])
        embedding = np.array(response.data[0].embedding)
        with cache_lock:
            embedding_cache[text_hash] = embedding
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
    save_embedding_cache()
    return chunk_embeddings

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)

def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1

def retrieve_relevant_chunks(query: str, chunk_embeddings: List[Tuple[str, np.ndarray]], top_n: int = 5) -> str:
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

def get_neo4j_context(user_query: str, limit: int = 100, retries: int = 3) -> List[Dict]:
    # Extrahiere Suchbegriff und Jahr aus der Abfrage
    keywords = user_query.lower().split()
    year = None
    for word in keywords:
        if word.isdigit() and len(word) == 4:
            year = int(word)
            keywords.remove(word)
            break
    search_term = " ".join(keywords)

    for attempt in range(retries):
        try:
            with driver.session() as session:
                context = []
                max_tokens = 8000
                result = session.run("""
                    MATCH (fm:FinancialMetric)-[:HAS_FINANCIAL_METRIC]->(entity)
                    WHERE (entity:ParentCompany OR entity:Company)
                    AND fm.value IS NOT NULL AND fm.year IS NOT NULL
                    AND toLower(fm.name) CONTAINS $search_term
                    AND ($year IS NULL OR fm.year = $year)
                    RETURN fm.name AS name, fm.year AS year, fm.value AS value, fm.unit AS unit,
                           fm.category AS category, entity.name AS entity_name
                    LIMIT $limit
                """, {
                    "search_term": search_term,
                    "year": year,
                    "limit": limit
                })

                for record in result:
                    # Überprüfe, ob alle erforderlichen Felder vorhanden sind
                    if (record["name"] is None or record["year"] is None or
                        record["value"] is None or record["unit"] is None or
                        record["entity_name"] is None):
                        logger.warning(f"⚠️ Unvollständige Daten in Neo4j: {dict(record)}")
                        continue  # Überspringe unvollständige Datensätze
                    context.append({
                        "name": str(record["name"]),
                        "year": record["year"],
                        "value": record["value"],
                        "unit": str(record["unit"]),
                        "category": str(record["category"]) if record["category"] else None,
                        "entity_name": str(record["entity_name"])
                    })

                if not context:
                    logger.warning("⚠️ Keine relevanten Daten in Neo4j gefunden")
                    return [{"message": "Keine ausreichenden Daten gefunden."}]
                
                token_count = estimate_tokens(str(context))
                if token_count > max_tokens:
                    context = context[:max(1, int(len(context) * max_tokens / token_count))]
                    logger.warning("⚠️ Neo4j-Kontext gekürzt")

                return context
        except ServiceUnavailable as e:
            logger.error(f"❌ Neo4j-Verbindungsfehler (Versuch {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)  # Exponentielles Backoff
                continue
            return [{"message": f"Fehler beim Laden der Neo4j-Daten: {e}"}]
        except Exception as e:
            logger.error(f"❌ Fehler beim Laden der Neo4j-Daten: {e}")
            return [{"message": f"Fehler beim Laden der Neo4j-Daten: {e}"}]

def generate_response(context: List[Dict], user_query: str) -> List[Dict]:
    if not context or (len(context) == 1 and "message" in context[0]):
        return [{"message": context[0]["message"] if "message" in context[0] else "Keine ausreichenden Daten gefunden."}]
    
    max_tokens = 12000
    context_text = "\n".join([f"{item['name']} ({item['entity_name']}, {item['year']}{', ' + item['category'] if item['category'] else ''}): {item['value']} {item['unit']}" for item in context])
    if estimate_tokens(context_text) > max_tokens:
        context_text = context_text[:max(1, int(len(context_text) * max_tokens / estimate_tokens(context_text)))]
        logger.warning("⚠️ Gesamter Kontext gekürzt")

    # Überprüfe, ob alle Einträge vollständig sind
    filtered_context = []
    for item in context:
        if all(key in item and item[key] is not None for key in ["name", "year", "value", "unit", "entity_name"]):
            filtered_context.append(item)
        else:
            logger.warning(f"⚠️ Unvollständiger Datensatz übersprungen: {item}")

    if not filtered_context:
        return [{"message": "Keine vollständigen Daten gefunden."}]

    # Wenn nur ein Ergebnis vorliegt, direkt zurückgeben
    if len(filtered_context) == 1:
        result = filtered_context[0]
        return [{"name": result["name"], "year": result["year"], "value": result["value"], "unit": result["unit"], "category": result["category"], "entity_name": result["entity_name"]}]

    # Wenn mehrere Ergebnisse vorliegen, Kontext direkt zurückgeben (Frontend übernimmt die Nachfrage)
    return filtered_context

# Eingabemodell für API
class Query(BaseModel):
    question: str

# Initialisierung beim Start der API
@app.on_event("startup")
async def startup_event():
    global documents, chunk_embeddings
    if not verify_neo4j_connection():
        logger.error("❌ Neo4j-Verbindung fehlgeschlagen, API startet trotzdem")
    load_embedding_cache()
    try:
        download_drive_folder(DOWNLOAD_PATH)
        documents = read_folder_data(DOWNLOAD_PATH, password="")
        if not documents:
            logger.warning("⚠️ Keine gültigen PDF-Dokumente gefunden, API startet ohne Dokumenten-Kontext")
        else:
            chunk_embeddings = create_embeddings_parallel(documents, max_length=300)
            if not chunk_embeddings:
                logger.warning("⚠️ Keine Embeddings erstellt, API startet ohne Dokumenten-Kontext")
    except Exception as e:
        logger.error(f"❌ Fehler beim Laden der Dokumente: {e}")

# API-Endpunkt
@app.post("/analyze")
async def analyze(query: Query):
    try:
        graph_context = get_neo4j_context(query.question, limit=100)
        document_context = retrieve_relevant_chunks(query.question, chunk_embeddings, top_n=5)
        combined_context = f"Neo4j-Daten:\n{json.dumps(graph_context, ensure_ascii=False)}\n\nGeschäftsberichte:\n{document_context}"
        answer = generate_response(graph_context, query.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"❌ Fehler bei der Verarbeitung: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Verwende die Umgebungsvariable PORT oder fallback auf 8000
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
