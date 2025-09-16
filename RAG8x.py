# main2.py
import os
import re
import tempfile
import urllib.request
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ---- LLM & Neo4j
from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.prompts import PromptTemplate

# ---- RAG / PDF
import fitz  # PyMuPDF
from chromadb import PersistentClient
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# ==================== Setup ====================
load_dotenv()

app = FastAPI(title="Graph-QA (Neo4j) + RAG PDF – Multi-Turn + Disambiguation")

# CORS – offen für Render + CodePen
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8001",
        "http://localhost:8001",
        "https://codepen.io",
        "https://cdpn.io",
        "*",  # für Tests
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- ENV
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")  # optional

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Vektorstore (lokal, wenn Render ohne Disk: ./chroma; mit Persistent Disk: /data/chroma)
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")
PDF_COLLECTION = os.getenv("PDF_COLLECTION", "siemens_2024")

# Embeddings (achte auf EMBED_MODEL vs EMB_MODEL)
EMBED_MODEL = os.getenv("EMBED_MODEL") or os.getenv("EMB_MODEL") or "text-embedding-3-small"

# Auto-Index: Lokaler Pfad oder Direkt-Download-URL (Drive: uc?export=download&id=…)
AUTO_PDF_PATH = os.getenv("AUTO_PDF_PATH")
AUTO_PDF_URL  = os.getenv("AUTO_PDF_URL")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY ist nicht gesetzt (.env)")

# Neo4j connect
graph = (
    Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD, database=NEO4J_DATABASE)
    if NEO4J_DATABASE
    else Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
)

# Schema (optional)
try:
    schema_text = graph.schema
except Exception:
    try:
        graph.refresh_schema()
        schema_text = graph.schema
    except Exception:
        schema_text = "(Schema konnte nicht gelesen werden – Whitelist ist bindend.)"

# ======= Erlaubte Labels/Beziehungen & Mappings =======
LABELS_ALLOW = [
    "Konzernmutter", "Tochterunternehmen", "Assoziierte_Gemeinschafts_Unternehmen",
    "Sonstige_Beteiligungen", "Periodenkennzahl", "Erfolgskennzahl",
    "Bestandskennzahl", "Nachhaltigkeitskennzahl", "Geschaeftsjahr",
    "Einheit", "Geschaeftsbereiche", "Stadt", "Land", "Rechnungslegungsstandard"
]
RELS_ALLOW = [
    "istTochterVon", "hatKonzernmutter", "beziehtSichAufPeriode", "beziehtSichAufUnternehmen",
    "hatFinanzkennzahl", "erstelltNachStandard", "ausgedruecktInEinheit",
    "segmentiertNachGeschaeftsbereich", "hatStandort", "liegtInLand"
]
ARRAY_PROPS = ["KennzahlWert", "anteilProzent"]

HOLDING_LABELS = ["Tochterunternehmen","Assoziierte_Gemeinschafts_Unternehmen","Sonstige_Beteiligungen"]
METRIC_LABELS  = ["Periodenkennzahl","Erfolgskennzahl","Bestandskennzahl","Nachhaltigkeitskennzahl"]

# Aliasse für deterministische Kennzahl+Jahr-Erkennung
METRIC_ALIASES = {
    "umsatzerlöse": "/Umsatzerlöse",
    "umsatz": "/Umsatzerlöse",
    "auftragseingang": "/Auftragseingang",
}

# ======== LLM =========
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

FEWSHOTS = """
Beispiel 1
Frage: Welche Tochterunternehmen hat die Siemens AG?
Hinweis: 'Tochterunternehmen' umfasst assoziierte Gemeinschaftsunternehmen und sonstige Beteiligungen.
Cypher:
MATCH (e:Tochterunternehmen|Assoziierte_Gemeinschafts_Unternehmen|Sonstige_Beteiligungen)-[:istTochterVon|hatKonzernmutter]->(m:Konzernmutter)
WHERE m.uri ENDS WITH '/Siemens_AG'
RETURN e.uri AS uri, e.anteilProzent[0] AS wert
ORDER BY uri;

Beispiel 2
Frage: Wie hoch war der Auftragseingang 2024 bei Siemens?
Cypher:
MATCH (k:Periodenkennzahl)-[:beziehtSichAufPeriode]->(p:Geschaeftsjahr),
      (k)-[:hatFinanzkennzahl]->(u:Konzernmutter)
WHERE u.uri ENDS WITH '/Siemens_AG'
  AND p.uri ENDS WITH '#2024'
  AND k.uri ENDS WITH '/Auftragseingang'
RETURN k.uri AS uri, k.KennzahlWert[0] AS wert;

Beispiel 3
Frage: Was kannst du mir zur Berliner Vermögensverwaltung GmbH sagen?
Cypher:
MATCH (n:Tochterunternehmen|Assoziierte_Gemeinschafts_Unternehmen|Sonstige_Beteiligungen)
WHERE n.uri ENDS WITH '/berliner_vermögensverwaltung_gmbh'
OPTIONAL MATCH (n)-[:istTochterVon|:hatKonzernmutter]->(m:Konzernmutter)
RETURN n.uri AS uri, n.anteilProzent[0] AS wert, n.Kommentar AS kommentar, labels(n) AS labels, m.uri AS konzern;
"""

CYPHER_GENERATION_TEMPLATE = """
Du übersetzt Benutzerfragen in **gültige Cypher**-Abfragen gegen Neo4j.
Verlauf (nur Kontext, nichts wiederholen):
{history}

DB-Schema:
{schema}

Regeln:
- Nur Labels: {labels_allow}
- Nur Relationen: {rels_allow}
- OWL/Schema-Kanten ignorieren (Class, ObjectProperty, DatatypeProperty, subClassOf, domain, range, Restriction, onProperty, onClass, members, first, rest, onDatatype).
- **Nie :Resource oder :NamedIndividual** in MATCH.
- "Tochterunternehmen" = :Tochterunternehmen|:Assoziierte_Gemeinschafts_Unternehmen|Sonstige_Beteiligungen; zum Konzern: [:istTochterVon|hatKonzernmutter].
- Periodenkennzahlen: (k)-[:beziehtSichAufPeriode]->(p:Geschaeftsjahr) und (k)-[:hatFinanzkennzahl]->(u:Konzernmutter).
- Jahr über p.uri (ENDS WITH '#YYYY')
- Node-Match via uri/… (ENDS WITH '/…' etc.)
- Zahlenarrays immer [0] (z. B. KennzahlWert[0], anteilProzent[0])
- **Immer** auch die `uri` der gefundenen Kennzahl/Unternehmens zurückgeben (`... AS uri`) – zusätzlich zu `wert` falls relevant.
- Liefere **nur** die Cypher-Query, ohne Markdown.

{fewshots}
Frage: {question}
"""

cypher_prompt = PromptTemplate(
    input_variables=["history", "schema", "labels_allow", "rels_allow", "fewshots", "question"],
    template=CYPHER_GENERATION_TEMPLATE
)

# ================= Helpers =================

def prettify_tail(uri: str) -> str:
    tail = uri.rsplit("/", 1)[-1]
    return re.sub(r"\s+", " ", tail.replace("_"," ").replace("%20"," ")).strip()

def is_company_question(text: str) -> bool:
    t = " " + text.lower() + " "
    if "tochter" in t or "beteilig" in t:
        return True
    COMPANY_HINTS = [
        " gmbh", " ag", " se", " kg", " kgaa", " ug", " ltd", " b.v", " s.r.l", " s.r.o", " s.a.", " spa",
        " holding", " gruppe", " & co. kg", " co. kg", " llc", " limited"
    ]
    return any(h in t for h in COMPANY_HINTS)

# ---------- Cypher Sanitizer ----------
def remove_generic_labels(cypher: str) -> str:
    cypher = re.sub(r":NamedIndividual\b", "", cypher)
    cypher = re.sub(r":Resource\b", "", cypher)
    return cypher.replace("::", ":")

def expand_holdings(cypher: str) -> str:
    cypher = cypher.replace(":Tochterunternehmen", ":Tochterunternehmen|Assoziierte_Gemeinschafts_Unternehmen|Sonstige_Beteiligungen")
    cypher = cypher.replace("[:istTochterVon]", "[:istTochterVon|hatKonzernmutter]")
    cypher = cypher.replace("[:hatKonzernmutter]", "[:istTochterVon|hatKonzernmutter]")
    return cypher

def expand_metric_labels(cypher: str) -> str:
    return cypher.replace(":Periodenkennzahl", ":Periodenkennzahl|Erfolgskennzahl|Bestandskennzahl|Nachhaltigkeitskennzahl")

def expand_uri_endswiths(cypher: str) -> str:
    def repl(m: re.Match) -> str:
        var = m.group("var")
        lit = m.group("lit")
        tail = lit.rsplit("/",1)[-1]
        cand = set()
        bases = {
            tail,
            tail.replace(" ","_"),
            tail.replace(".","_").replace("-","_"),
            tail.lower(),
            tail.lower().replace(" ","_").replace(".","_").replace("-","_")
        }
        for t in bases:
            cand.add(f"toLower({var}.uri) ENDS WITH toLower('/{t}')")
            cand.add(f"toLower({var}.uri) ENDS WITH toLower('{t}')")
            # KORREKTE Variante (die mit den Klammern passt):
            cand.add(f"toLower(replace(replace({var}.uri,'-','_'),'.','_')) CONTAINS toLower('{t}')")
        return "(" + " OR ".join(sorted(cand)) + ")"
    pat = re.compile(r"(?P<var>[A-Za-z_][A-Za-z0-9_]*)\.uri\s+ENDS\s+WITH\s+'(?P<lit>[^']+)'", re.IGNORECASE)
    return pat.sub(repl, cypher)

def expand_year_filters(cypher: str) -> str:
    pat1 = re.compile(r"(p\.uri\s*ENDS\s*WITH\s*'#(\d{4})')", re.IGNORECASE)
    cypher = pat1.sub(lambda m: f"({m.group(1)} OR p.uri ENDS WITH '/{m.group(2)}' OR toLower(k.uri) CONTAINS '_{m.group(2)}')", cypher)
    return cypher

def sanitize_and_fix(cypher: str, user_question: str = "") -> str:
    lowered = " " + cypher.lower().replace("\n", " ") + " "
    if any(kw in lowered for kw in [" create "," merge "," delete "," remove "," set ",
                                    " call dbms", " apoc.trigger", " apoc.schema", " load csv",
                                    " create constraint", " drop constraint", " create index", " drop index"]):
        raise HTTPException(status_code=400, detail="Nur Leseabfragen sind erlaubt.")
    fixed = cypher.strip()
    fixed = remove_generic_labels(fixed)
    for prop in ["KennzahlWert", "anteilProzent"]:
        pattern = re.compile(r"(\b[A-Za-z_][A-Za-z0-9_]*)\." + re.escape(prop) + r"(?!\s*\[)")
        fixed = pattern.sub(r"coalesce(\1." + prop + r"[0], \1." + prop + r")", fixed)
    if not is_company_question(user_question):
        fixed = expand_metric_labels(fixed)
    fixed = expand_holdings(fixed)
    fixed = expand_uri_endswiths(fixed)
    fixed = expand_year_filters(fixed)
    fixed = re.sub(r"\bAS\s+anteil\b", "AS wert", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"\bAS\s+(value|betrag|amount)\b", "AS wert", fixed, flags=re.IGNORECASE)
    return fixed

def de_format_number(x: Any) -> str:
    try:
        s = f"{float(x):,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return s
    except Exception:
        return str(x)

def _tail(uri: Optional[str]) -> Optional[str]:
    return uri.rsplit("/",1)[-1] if uri else None

def short_name(uri: Optional[str]) -> Optional[str]:
    return prettify_tail(uri) if uri else None

def std_short(std_uri: Optional[str]) -> Optional[str]:
    if not std_uri: return None
    s = prettify_tail(std_uri).upper()
    if "IFRS" in s: return "IFRS"
    if "HGB" in s: return "HGB"
    return s

def _norm(s: str) -> str:
    return s.lower().replace("ä","ae").replace("ö","oe").replace("ü","ue").strip()

METRIC_RE = re.compile(
    r"\b(umsatzerl(ö|oe)se|umsatz|auftragseingang)\b.*?(20\d{2})",
    re.IGNORECASE
)

def parse_metric_year_question(q: str) -> Optional[Dict[str, Any]]:
    m = METRIC_RE.search(q or "")
    if not m:
        return None
    metric_raw = _norm(m.group(1))
    year = m.group(3)
    for key, tail in METRIC_ALIASES.items():
        if _norm(key) in metric_raw:
            return {"tail": tail, "year": year}
    return None

# ================= RAG / PDF =================

# Chroma mit Telemetrie aus
os.makedirs(CHROMA_DIR, exist_ok=True)
chroma = PersistentClient(path=CHROMA_DIR, settings=ChromaSettings(anonymized_telemetry=False))
openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY, model_name=EMBED_MODEL)

def _get_collection():
    return chroma.get_or_create_collection(name=PDF_COLLECTION, embedding_function=openai_ef)

def _pdf_to_chunks(pdf_path: str) -> List[Dict[str, Any]]:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text") or ""
        if text.strip():
            pages.append({"page": i+1, "text": text})
    doc.close()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200, separators=["\n\n", "\n", ". ", " "])
    chunks: List[Dict[str, Any]] = []
    for p in pages:
        for c in splitter.split_text(p["text"]):
            chunks.append({
                "id": f"{p['page']}-{abs(hash(c))}",
                "text": c,
                "meta": {"source": os.path.basename(pdf_path) if pdf_path else "PDF", "page": p["page"]}
            })
    return chunks

def _ingest_pdf(pdf_path: str) -> int:
    col = _get_collection()
    chunks = _pdf_to_chunks(pdf_path)
    if not chunks:
        return 0
    col.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["meta"] for c in chunks],
    )
    return len(chunks)

def _collection_count() -> int:
    try:
        col = _get_collection()
        return col.count()
    except Exception:
        return 0

def _download_to_tmp(url: str) -> str:
    """Lädt eine Datei nach /tmp und gibt den Pfad zurück."""
    if not url:
        raise ValueError("URL fehlt")
    fd, tmp_path = tempfile.mkstemp(prefix="ragpdf_", suffix=".pdf")
    os.close(fd)
    urllib.request.urlretrieve(url, tmp_path)
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        raise RuntimeError("Download fehlgeschlagen oder Datei leer")
    return tmp_path

def _query_rag(question: str, top_k: int = 6) -> List[Dict[str, Any]]:
    """Embedding + BM25 Kombi; erst Embedding-Retrieval, dann BM25-Re-Rank."""
    col = _get_collection()
    q = col.query(query_texts=[question], n_results=top_k*2)
    docs = q.get("documents", [[]])[0]
    metas = q.get("metadatas", [[]])[0]
    if not docs:
        return []
    bm = BM25Okapi([d.split() for d in docs])
    scores = bm.get_scores(question.split())
    pairs = list(zip(docs, metas, scores))
    pairs.sort(key=lambda x: x[2], reverse=True)
    pairs = pairs[:top_k]
    return [{"text": d, "meta": m} for d, m, _ in pairs]

# ---- Reranking & Superlativ-Helfer ----
REGION_KEYWORDS = [
    "europa, gus, afrika, naher und mittlerer osten",
    "amerika", "asien, australien",
    "sitz des kunden", "auftrags", "umsatzerlöse", "umsatz", "region"
]

def _contains_number(s: str) -> bool:
    return bool(re.search(r"\d", s))

def _number_density(s: str) -> float:
    digits = len(re.findall(r"\d", s))
    formatted = len(re.findall(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", s))
    return digits + 8*formatted

def _keyword_boost(s: str, kws: List[str]) -> float:
    t = s.lower()
    return sum(5.0 for k in kws if k in t)

def _is_superlative_question(q: str) -> bool:
    ql = q.lower()
    triggers = ["höchste", "höchsten", "größte", "groesste", "max", "top", "am meisten", "höchster", "höchstes"]
    return any(w in ql for w in triggers)

def _want_revenue(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ["umsatz", "umsatzerlöse", "umsatzerloese"])

def _want_order_intake(q: str) -> bool:
    ql = q.lower()
    return "auftragseingang" in ql

def _rerank_for_tables(question: str, ctx: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    needs_max = _is_superlative_question(question)
    want_rev = _want_revenue(question)
    want_oi = _want_order_intake(question)
    rescored = []
    for c in ctx:
        text = c.get("text") or ""
        score = 0.0
        score += _number_density(text) * (3.0 if needs_max else 1.5)
        score += _keyword_boost(text, REGION_KEYWORDS)
        if want_rev and ("umsatzerlöse" in text.lower() or "umsatz" in text.lower()):
            score += 10.0
        if want_oi and ("auftrags" in text.lower()):
            score += 10.0
        rescored.append((score, c))
    rescored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in rescored]

# ---- Parser für Regionstabellen ----
REGION_ALIASES = {
    "europa": "Europa, GUS, Afrika, Naher und Mittlerer Osten",
    "amerika": "Amerika",
    "asien": "Asien, Australien",
}

def _norm_de_number(num_str: str) -> float:
    s = num_str.strip().replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return float("nan")

def _extract_region_values_from_text(text: str, need_revenue: bool, need_order: bool) -> Dict[str, float]:
    t = text.lower()
    if need_revenue and ("umsatzerlöse" not in t and "umsatz" not in t):
        return {}
    if need_order and ("auftrag" not in t):
        return {}
    values: Dict[str, float] = {}
    for raw in text.splitlines():
        l = raw.lower()
        if "darin:" in l:
            continue
        region_key = None
        if "europa" in l and ("naher" in l or "mittlerer" in l or "gus" in l or "afrika" in l):
            region_key = "europa"
        elif "amerika" in l and "latin" not in l:
            region_key = "amerika"
        elif "asien" in l and "australien" in l:
            region_key = "asien"
        if not region_key:
            continue
        m = re.search(r"(\d{1,3}(?:\.\d{3})+(?:,\d+)?)", raw)
        if not m:
            continue
        val = _norm_de_number(m.group(1))
        if val != val:
            continue
        canonical = REGION_ALIASES[region_key]
        values[canonical] = val
    return values

# ====== Lifecycle ======
@app.on_event("startup")
def _ensure_rag_ready():
    try:
        count_before = _collection_count()
        # 1) Lokaler Pfad
        if AUTO_PDF_PATH and os.path.exists(AUTO_PDF_PATH) and count_before == 0:
            n = _ingest_pdf(AUTO_PDF_PATH)
            print(f"[RAG] Lokale PDF indiziert: {AUTO_PDF_PATH} (Chunks: {n})")
        # 2) Remote URL (z. B. Google Drive "uc?export=download&id=…")
        elif AUTO_PDF_URL and count_before == 0:
            try:
                tmp_pdf = _download_to_tmp(AUTO_PDF_URL)
                n = _ingest_pdf(tmp_pdf)
                print(f"[RAG] Ingest aus URL abgeschlossen: {n} Chunks. (Quelle: AUTO_PDF_URL)")
                try:
                    os.remove(tmp_pdf)
                except Exception:
                    pass
            except Exception as e:
                print(f"[RAG] Download/Index von AUTO_PDF_URL fehlgeschlagen: {e}")
        else:
            print(f"[RAG] Sammlung '{PDF_COLLECTION}' bereits befüllt (Count={count_before}) oder keine Quelle gesetzt.")
    except Exception as e:
        print(f"[RAG] Startup-Fehler: {e}")

class RAGStatus(BaseModel):
    collection: str
    count: int
    auto_pdf: bool
    auto_pdf_path: Optional[str]

@app.get("/rag/status", response_model=RAGStatus)
def rag_status():
    return RAGStatus(
        collection=PDF_COLLECTION,
        count=_collection_count(),
        auto_pdf=bool(AUTO_PDF_PATH and os.path.exists(AUTO_PDF_PATH)),
        auto_pdf_path=AUTO_PDF_PATH if AUTO_PDF_PATH and os.path.exists(AUTO_PDF_PATH) else None
    )

@app.post("/rag/ingest_pdf")
def rag_ingest_pdf(file_path: str):
    if not os.path.exists(file_path):
        raise HTTPException(400, f"Datei nicht gefunden: {file_path}")
    n = _ingest_pdf(file_path)
    return {"ingested": n}

@app.post("/rag/ingest_pdf_url")
def rag_ingest_pdf_url(url: str):
    """Manuelles Ingesten einer Remote-PDF-URL."""
    try:
        tmp_pdf = _download_to_tmp(url)
        n = _ingest_pdf(tmp_pdf)
        try:
            os.remove(tmp_pdf)
        except Exception:
            pass
        return {"ingested": n, "source": url}
    except Exception as e:
        raise HTTPException(400, f"Download/Index fehlgeschlagen: {e}")

# ================= API =================

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/schema")
def get_schema():
    return {
        "schema": schema_text,
        "labels_allow": LABELS_ALLOW,
        "rels_allow": RELS_ALLOW,
        "array_props": ARRAY_PROPS,
        "holding_labels": HOLDING_LABELS,
        "metric_labels": METRIC_LABELS,
        "metric_aliases": METRIC_ALIASES
    }

class Message(BaseModel):
    role: str
    content: str

class ChatBody(BaseModel):
    messages: List[Message]
    source_mode: Optional[str] = "auto"

def _history_text(msgs: List[Message]) -> str:
    return "\n".join([f"{m.role}: {m.content}" for m in msgs][-8:])

# ---- Quick List for UI ----
@app.get("/list")
def list_items(kind: str = Query(..., pattern="^(company|metric)$"), q: str = "", limit: int = 5000):
    if kind == "company":
        rows = graph.query(
            """
            MATCH (n)
            WHERE ('Konzernmutter' IN labels(n)) OR ANY(l IN labels(n) WHERE l IN $holding)
            RETURN n.uri AS uri
            ORDER BY toLower(n.uri)
            LIMIT $limit
            """,
            params={"holding": HOLDING_LABELS, "limit": limit}
        )
        items = [{"uri": r["uri"], "label": prettify_tail(r["uri"])} for r in rows]
    else:
        rows = graph.query(
            """
            MATCH (k)
            WHERE ANY(l IN labels(k) WHERE l IN $ml)
            RETURN k.uri AS uri
            ORDER BY toLower(k.uri)
            LIMIT $limit
            """,
            params={"ml": METRIC_LABELS, "limit": limit}
        )
        items = [{"uri": r["uri"], "label": prettify_tail(r["uri"])} for r in rows]
    if q:
        ql = q.lower()
        items = [it for it in items if ql in (it["label"] or "").lower()]
    return {"items": items}

# ---- Detailauflösung ----
@app.get("/valueByUri")
def value_by_uri(uri: str = Query(..., description="Exakte URI eines Knotens (Kennzahl oder Unternehmen)")):
    rows = graph.query("""
        MATCH (n {uri: $uri})
        WITH n, labels(n) AS labs
        OPTIONAL MATCH (n)-[:beziehtSichAufPeriode]->(p:Geschaeftsjahr)
        OPTIONAL MATCH (n)-[:beziehtSichAufUnternehmen|hatFinanzkennzahl]-(u:Konzernmutter)
        OPTIONAL MATCH (n)-[:ausgedruecktInEinheit]->(e:Einheit)
        OPTIONAL MATCH (n)-[:erstelltNachStandard]->(std:Rechnungslegungsstandard)
        OPTIONAL MATCH (n)-[:hatStandort]->(s:Stadt)-[:liegtInLand]->(l:Land)
        RETURN labs AS labels,
               n.uri AS uri,
               coalesce(n.KennzahlWert[0], n.KennzahlWert, n.anteilProzent[0], n.anteilProzent) AS wert,
               p.uri AS periode,
               u.uri AS konzern,
               e.uri AS einheit,
               std.uri AS standard,
               s.uri AS stadt,
               l.uri AS land,
               n.Kommentar AS kommentar
        LIMIT 1
    """, params={"uri": uri})
    if not rows:
        raise HTTPException(404, "Nicht gefunden")
    r = rows[0]
    labs = r["labels"] or []
    kind = "metric" if any(l in METRIC_LABELS for l in labs) else ("company" if any((l in HOLDING_LABELS) or (l == "Konzernmutter") for l in labs) else "other")
    return {
        "uri": r["uri"],
        "label": prettify_tail(r["uri"]),
        "wert": r["wert"],
        "periode": r["periode"],
        "konzern": r["konzern"],
        "einheit": r["einheit"],
        "standard": r.get("standard"),
        "stadt": r.get("stadt"),
        "land": r.get("land"),
        "kommentar": r.get("kommentar"),
        "kind": kind,
    }

# ---- Graph-Ego ----
@app.get("/graph/ego")
def graph_ego(uri: str, depth: int = 1, limit: int = 60):
    if depth < 1 or depth > 2:
        depth = 1

    q = f"""
    MATCH (center {{uri: $uri}})
    OPTIONAL MATCH p=(center)-[r*1..{depth}]-(m)
    WITH center, collect(p) AS ps
    WITH center,
         reduce(ns = [], p IN ps | ns + nodes(p)) AS ns,
         reduce(rs = [], p IN ps | rs + relationships(p)) AS rs
    WITH center,
         CASE WHEN size(ns)=0 THEN [center] ELSE ns END AS ns,
         rs
    WITH [n IN ns | n] AS nodes, [x IN rs | x] AS rels
    RETURN
      [n IN nodes | {{uri: n.uri, labels: labels(n)}}] AS nodes,
      [r IN rels  | {{source: startNode(r).uri, target: endNode(r).uri, type: type(r)}}] AS edges
    LIMIT $limit
    """
    rows = graph.query(q, params={"uri": uri, "limit": limit})
    if not rows:
        raise HTTPException(404, "Kein Graph für diese URI gefunden.")

    data = rows[0]
    raw_nodes = data.get("nodes") or []
    raw_edges = data.get("edges") or []

    out_nodes = []
    seen = set()
    for n in raw_nodes:
        uid = n.get("uri")
        if not uid or uid in seen:
            continue
        seen.add(uid)
        labs = n.get("labels") or []
        label = labs[0] if labs else "Node"
        cap = prettify_tail(uid)
        out_nodes.append({"id": uid, "label": label, "caption": cap})

    out_edges = []
    for e in raw_edges:
        s = e.get("source"); t = e.get("target"); typ = e.get("type") or "REL"
        if s and t:
            out_edges.append({"source": s, "target": t, "type": typ})

    return {"nodes": out_nodes, "edges": out_edges}

# ---- /chat_plus ----
@app.post("/chat_plus")
def chat_plus(body: ChatBody):
    if not body.messages:
        raise HTTPException(400, "messages ist leer")

    mode = (body.source_mode or "auto").lower()
    force_pdf = (mode == "pdf")
    force_graph = (mode == "graph")

    user_msgs = [m.content for m in body.messages if m.role == "user"]
    question = user_msgs[-1] if user_msgs else body.messages[-1].content
    history_text = _history_text(body.messages[:-1])

    cypher_raw = None
    cypher_exec = None
    rows: List[Dict[str, Any]] = []

    # ---- deterministischer Kennzahl+Jahr Pfad (Graph) – vor LLM
    if not force_pdf:
        my = parse_metric_year_question(question)
        if my:
            tail = my["tail"]
            year = my["year"]
            cypher_exec = f"""
            MATCH (k)-[:beziehtSichAufPeriode]->(p:Geschaeftsjahr)
            OPTIONAL MATCH (k)-[:hatFinanzkennzahl]->(u:Konzernmutter)
            WHERE (toLower(k.uri) ENDS WITH toLower('{tail}')
                   OR toLower(replace(replace(k.uri,'-','_'),'.','_')) CONTAINS toLower('{tail}'))
              AND (p.uri ENDS WITH '#{year}' OR p.uri ENDS WITH '/{year}')
            RETURN k.uri AS uri, coalesce(k.KennzahlWert[0], k.KennzahlWert) AS wert
            ORDER BY uri
            LIMIT 1
            """.strip()
            try:
                rows = graph.query(cypher_exec)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Cypher-Ausführung fehlgeschlagen: {e}")

            if rows:
                r0 = rows[0]
                uri = r0.get("uri")
                if uri:
                    try:
                        det = value_by_uri(uri)
                        if det["kind"] == "metric":
                            value = det.get("wert")
                            year_txt = (short_name(det.get("periode")) or "").split("#")[-1] or "?"
                            einheit = short_name(det.get("einheit")) or ""
                            std = std_short(det.get("standard")) or "IFRS/HGB"
                            val_txt = de_format_number(value)
                            label = det["label"]
                            text = f'Die Kennzahl "{label}" wird nach {std} ermittelt und belief sich im Geschäftsjahr {year_txt} auf {val_txt} {einheit}.'
                            return {
                                "mode": "answer",
                                "cypher": "(deterministisch)",
                                "cypher_executed": cypher_exec,
                                "rows": rows,
                                "answer": text
                            }
                    except Exception:
                        pass
                if "wert" in r0 and r0["wert"] is not None:
                    return {
                        "mode": "answer",
                        "cypher": "(deterministisch)",
                        "cypher_executed": cypher_exec,
                        "rows": rows,
                        "answer": f"Ergebnis: {de_format_number(r0['wert'])}"
                    }
            # wenn keine Rows → normal weiter

    # -------- Graph (LLM-Cypher), wenn nicht "pdf only"
    if not force_pdf:
        cypher_in = cypher_prompt.format(
            history=history_text or "(kein Verlauf)",
            schema=schema_text,
            labels_allow=", ".join(LABELS_ALLOW),
            rels_allow=", ".join(RELS_ALLOW),
            fewshots=FEWSHOTS,
            question=question
        )
        cypher_raw = llm.invoke(cypher_in).content.strip().strip("`")
        cypher_exec = sanitize_and_fix(cypher_raw, question)
        try:
            rows = graph.query(cypher_exec)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Cypher-Ausführung fehlgeschlagen: {e}")

    # ---- Disambiguation & formatierte Antworten nur, wenn Graph benutzt wird
    if not force_pdf:
        # Disambiguation: Firmen
        if (len(rows) > 1 or (not rows and is_company_question(question))):
            if is_company_question(question):
                cq = """
                MATCH (n)
                WHERE ( 'Konzernmutter' IN labels(n) ) OR ANY(l IN labels(n) WHERE l IN $holding)
                WITH n, toLower(n.uri) AS u
                WHERE u CONTAINS toLower($needle)
                OPTIONAL MATCH (n)-[:istTochterVon|hatKonzernmutter]->(m:Konzernmutter)
                RETURN n.uri AS uri, m.uri AS konzern
                ORDER BY uri
                LIMIT 8
                """
                cands = graph.query(cq, params={"holding": HOLDING_LABELS, "needle": question})
                if cands:
                    options = [{"uri": r["uri"], "label": prettify_tail(r["uri"])} for r in cands]
                    return {
                        "mode": "clarify",
                        "question": "Ich habe mehrere passende Unternehmen gefunden. Welches meinst du?",
                        "options": options,
                        "cypher_tried": cypher_exec
                    }

        # Kennzahlen-Disambiguation
        if len(rows) > 1 and not is_company_question(question):
            opts = []
            for r in rows[:20]:
                uri = r.get("uri")
                if uri:
                    opts.append({"uri": uri, "label": prettify_tail(uri), "wert": r.get("wert")})
            if opts:
                return {
                    "mode": "clarify",
                    "question": "Ich habe mehrere passende Kennzahlen gefunden. Welche meinst du genau?",
                    "options": opts,
                    "cypher_tried": cypher_exec
                }

        # Formatierte Antworten
        if rows:
            r0 = rows[0]
            uri = r0.get("uri")
            if uri:
                try:
                    det = value_by_uri(uri)
                    if det["kind"] == "company":
                        anteil = det.get("wert")
                        anteil_txt = f"{de_format_number(anteil)} %" if anteil is not None else "100 %"
                        konzern_txt = short_name(det.get("konzern")) or "die Konzernmutter"
                        sitz = ""
                        if det.get("stadt") or det.get("land"):
                            city = short_name(det.get("stadt")) or "?"
                            country = short_name(det.get("land")) or "?"
                            sitz = f"\nSitz: {city}, {country}"
                        kommentar = det.get("kommentar")
                        kmt = f"\nKommentar: {kommentar}" if kommentar else ""
                        text = f"Die {konzern_txt} hält {anteil_txt} an {det['label']}.{sitz}{kmt}"
                        return {
                            "mode": "answer",
                            "cypher": cypher_raw,
                            "cypher_executed": cypher_exec,
                            "rows": rows,
                            "answer": text
                        }
                    elif det["kind"] == "metric":
                        value = det.get("wert")
                        year = short_name(det.get("periode"))
                        year = year.split("#")[-1] if year else "?"
                        einheit = short_name(det.get("einheit")) or ""
                        std = std_short(det.get("standard")) or "IFRS/HGB"
                        val_txt = de_format_number(value)
                        label = det["label"]
                        text = f'Die Kennzahl "{label}" wird nach {std} ermittelt und belief sich im Geschäftsjahr {year} auf {val_txt} {einheit}.'
                        return {
                            "mode": "answer",
                            "cypher": cypher_raw,
                            "cypher_executed": cypher_exec,
                            "rows": rows,
                            "answer": text
                        }
                except Exception:
                    pass

            # Fallback: Wert-only
            if "wert" in r0 and r0["wert"] is not None:
                return {
                    "mode": "answer",
                    "cypher": cypher_raw,
                    "cypher_executed": cypher_exec,
                    "rows": rows,
                    "answer": f"Ergebnis: {de_format_number(r0['wert'])}"
                }

    # ---- RAG (PDF) – wenn explizit (force_pdf) oder als Fallback
    should_use_rag = force_pdf or (not rows and not force_graph and not is_company_question(question))
    if should_use_rag:
        ctx = _query_rag(question, top_k=8)
        if ctx:
            # --- Re-Rank wie gehabt
            ctx = _rerank_for_tables(question, ctx)

            needs_max = _is_superlative_question(question)
            want_rev = _want_revenue(question)
            want_oi = _want_order_intake(question)

            # --- 1) deterministischer Pfad für Superlative (falls sinnvoll)
            if needs_max and (want_rev or want_oi):
                collected: Dict[str, float] = {}
                used_pages, sources = set(), set()
                for c in ctx:
                    txt = (c.get("text") or "").strip()
                    meta = c.get("meta") or {}
                    if not txt:
                        continue
                    vals = _extract_region_values_from_text(txt, need_revenue=want_rev, need_order=want_oi)
                    if vals:
                        for k, v in vals.items():
                            if (k not in collected) or (abs(v) > abs(collected[k])):
                                collected[k] = v
                    if "page" in meta:
                        used_pages.add(meta["page"])
                    if meta.get("source"):
                        sources.add(meta["source"])

                if collected:
                    best_key = max(collected.keys(), key=lambda k: abs(collected[k]))
                    best_val = collected[best_key]
                    metric_txt = "Umsatzerlöse" if want_rev else "Auftragseingang"
                    answer = f"{metric_txt}: höchste Region = {best_key} ({best_val:,.0f})".replace(",", ".")
                    source_name = ", ".join(sorted(sources)) if sources else None
                    return {
                        "mode": "answer",
                        "answer": answer + (f"\n\nQuelle: {source_name}" if source_name else ""),
                        "pdf_pages": [],
                        "pdf_source": source_name or "PDF"
                    }

                # keine eindeutigen Zahlen → defensiv
                sources = { (c.get("meta") or {}).get("source") for c in ctx if (c.get("meta") or {}).get("source") }
                source_name = ", ".join(sorted(sources)) if sources else None
                return {
                    "mode": "answer",
                    "answer": "Keine Daten gefunden." + (f"\n\nQuelle: {source_name}" if source_name else ""),
                    "pdf_pages": [],
                    "pdf_source": source_name or "PDF"
                }

            # --- 2) generischer RAG-Pfad
            by_page: Dict[int, List[str]] = {}
            sources = set()
            for c in ctx:
                meta = c.get("meta") or {}
                page = meta.get("page")
                if page is None:
                    continue
                txt = (c.get("text") or "").strip()
                if txt:
                    by_page.setdefault(page, []).append(txt)
                if meta.get("source"):
                    sources.add(meta["source"])

            excerpts = []
            for p in sorted(by_page.keys())[:5]:
                joined = " ".join(by_page[p])
                excerpt = re.sub(r"\s+", " ", joined).strip()
                excerpts.append(f"— Seite {p} —\n{excerpt[:1200]}")

            contexts_text = "\n\n".join(excerpts) if excerpts else ""
            source_name = ", ".join(sorted(sources)) if sources else None

            sys_rules = (
                "Du bist ein sehr präziser Assistent. Antworte ausschließlich mit Informationen aus dem Kontextauszug. "
                "Wenn eine Antwort im Kontext nicht belegt ist, antworte mit 'Keine Daten gefunden.' "
                "Liefere eine kurze, vollständige Antwort in ganzen Sätzen. "
                "Wenn konkrete Zahlen/Bezeichnungen vorhanden sind, übernimm sie exakt (inkl. Einheiten/Jahresangaben)."
            )
            if needs_max:
                sys_rules += (
                    " Bei Superlativen (z. B. 'höchste/größte') liefere nur den Eintrag mit dem größten absoluten Wert, "
                    "sofern im Kontext eindeutig Zahlen vorliegen; sonst: 'Keine Daten gefunden.'"
                )

            prompt = (
                f"{sys_rules}\n\n"
                f"Kontextauszüge:\n{contexts_text}\n\n"
                f"Frage: {question}\n\n"
                f"Anforderungen:\n- Antworte kurz.\n- Keine Spekulation.\n- Nutze nur den Kontext.\n\nAntwort:"
            )

            try:
                raw = llm.invoke(prompt)
                answer_text = (raw.content if hasattr(raw, "content") else str(raw)).strip()
            except Exception:
                answer_text = (llm.predict(prompt) if hasattr(llm, "predict") else "").strip()

            if not answer_text:
                answer_text = "Keine Daten gefunden."

            return {
                "mode": "answer",
                "answer": answer_text + (f"\n\nQuelle: {source_name}" if source_name else ""),
                "pdf_pages": [],
                "pdf_source": source_name or "PDF"
            }

    # Nichts gefunden
    return {"mode": "answer", "answer": "Keine Daten gefunden."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
