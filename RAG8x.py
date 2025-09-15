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

# Vektorstore
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma")
PDF_COLLECTION = os.getenv("PDF_COLLECTION", "siemens_2024")
EMBED_MODEL = os.getenv("EMBED_MODEL") or os.getenv("EMB_MODEL") or "text-embedding-3-small"

AUTO_PDF_PATH = os.getenv("AUTO_PDF_PATH")
AUTO_PDF_URL = os.getenv("AUTO_PDF_URL")

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
METRIC_LABELS = ["Periodenkennzahl","Erfolgskennzahl","Bestandskennzahl","Nachhaltigkeitskennzahl"]

# Aliasse (Auftragseingang bleibt Tail; Umsatzerlöse als Keyword)
METRIC_ALIASES = {
    "umsatzerlöse": "/Umsatzerlöse",
    "umsatz": "/Umsatzerlöse",
    "auftragseingang": "/Auftragseingang",
    "umsatzerlose": "/Umsatzerlöse",
    "umsatzerlöse_2023": "/Umsatzerlöse_2023",
    "umsatzerlose_2023": "/Umsatzerlöse_2023"
}

# Zusätzliche Aliasse für generische Kennzahlen (ohne Jahr)
METRIC_ALIASES.update({
    "cash conversion rate": "cash_conversion_rate",
    "ccr": "cash_conversion_rate",
    "bilanzgewinn": "bilanzgewinn",
    "ergebnis": "ergebnis",
    "ergebnismarge": "ergebnismarge",
    "roce": "roce",
    "ebit": "ebit",
})

# NEW: Outlook-Terme für Intent-Erkennung
OUTLOOK_TERMS = ["ausblick", "zukunft", "zukünft", "prognose", "erwart", "outlook", "guidance", "trend"]

def is_outlook_question(text: str) -> bool:
    t = " " + (text or "").lower() + " "
    return any(term in t for term in OUTLOOK_TERMS)

# ======== LLM =========
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, openai_api_key=OPENAI_API_KEY)

FEWSHOTS = """
Beispiel 1
Frage: Welche Tochterunternehmen hat die Siemens AG?
Hinweis: 'Tochterunternehmen' umfasst assoziierte Gemeinschaftsunternehmen und sonstige Beteiligungen.
Cypher:
MATCH (e:Tochterunternehmen|Assoziierte_Gemeinschafts_Unternehmen|Sonstige_Beteiligungen)-[:istTochterVon|hatKonzernmutter]->(m:Konzernmutter)
WHERE toLower(m.uri) ENDS WITH toLower('/siemens_ag')
RETURN e.uri AS uri, e.anteilProzent[0] AS wert
ORDER BY uri;

Beispiel 2
Frage: Wie hoch war der Auftragseingang 2023 bei Siemens?
Cypher:
MATCH (k:Periodenkennzahl|Erfolgskennzahl)-[:beziehtSichAufPeriode]->(p:Geschaeftsjahr),
      (k)-[:beziehtSichAufUnternehmen]->(u:Konzernmutter)
WHERE toLower(u.uri) ENDS WITH toLower('/siemens_ag')
  AND toLower(p.uri) ENDS WITH toLower('#2023')
  AND toLower(k.uri) ENDS WITH toLower('/auftragseingang_2023')
RETURN k.uri AS uri, k.KennzahlWert[0] AS wert;

Beispiel 2b
Frage: Wie hoch waren die Umsatzerlöse 2023 bei Siemens?
Cypher:
MATCH (k:Periodenkennzahl|Erfolgskennzahl)-[:beziehtSichAufPeriode]->(p:Geschaeftsjahr),
      (k)-[:beziehtSichAufUnternehmen|hatFinanzkennzahl]-(u:Konzernmutter)
WHERE u.uri ENDS WITH '/Siemens_AG'
  AND (p.uri ENDS WITH '#2023' OR p.uri ENDS WITH '/2023' OR toLower(k.uri) CONTAINS '_2023')
  AND (toLower(k.uri) CONTAINS 'umsatzerlöse' OR toLower(k.uri) CONTAINS 'umsatz')
RETURN k.uri AS uri, coalesce(k.KennzahlWert[0], k.KennzahlWert) AS wert, p.uri AS periode;

Beispiel 3
Frage: Was kannst du mir zur Berliner Vermögensverwaltung GmbH sagen?
Cypher:
MATCH (n:Tochterunternehmen|Assoziierte_Gemeinschafts_Unternehmen|Sonstige_Beteiligungen)
WHERE toLower(n.uri) ENDS WITH toLower('/berliner_vermögensverwaltung_gmbh')
RETURN n.uri AS uri, n.anteilProzent[0] AS wert
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
- "Tochterunternehmen" = :Tochterunternehmen|:Assoziierte_Gemeinschafts_Unternehmen|Sonstige_Beteiligungen; zum Konzern: [:istTochterVon|hatKonzernmutter].
- Periodenkennzahlen: (k)-[:beziehtSichAufPeriode]->(p:Geschaeftsjahr) und (k)-[:beziehtSichAufUnternehmen|:hatFinanzkennzahl]->(u:Konzernmutter).
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
    union = ":Periodenkennzahl|Erfolgskennzahl|Bestandskennzahl|Nachhaltigkeitskennzahl"
    for lab in ["Periodenkennzahl", "Erfolgskennzahl", "Bestandskennzahl", "Nachhaltigkeitskennzahl"]:
        cypher = re.sub(rf":{lab}\b", union, cypher)
    return cypher

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
        uml = (tail
               .replace("ä","ae").replace("Ä","Ae")
               .replace("ö","oe").replace("Ö","Oe")
               .replace("ü","ue").replace("Ü","Ue")
               .replace("ß","ss"))
        bases |= { uml, uml.lower(), uml.replace(" ","_").lower() }
        for t in bases:
            cand.add(f"toLower({var}.uri) ENDS WITH toLower('/{t}')")
            cand.add(f"toLower({var}.uri) ENDS WITH toLower('{t}')")
            cand.add(f"toLower(replace(replace({var}.uri,'-','_'),'.','_')) CONTAINS toLower('{t}')")
        return "(" + " OR ".join(sorted(cand)) + ")"
    pat = re.compile(r"(?P<var>[A-Za-z_][A-Za-z0-9_]*)\.uri\s+ENDS\s+WITH\s+'(?P<lit>[^']+)'", re.IGNORECASE)
    return pat.sub(repl, cypher)

def expand_year_filters(cypher: str) -> str:
    pat1 = re.compile(r"(p\.uri\s*ENDS\s*WITH\s*'#(\d{4})')", re.IGNORECASE)
    cypher = pat1.sub(lambda m: f"({m.group(1)} OR p.uri ENDS WITH '/{m.group(2)}' OR toLower(k.uri) CONTAINS '_{m.group(2)}')", cypher)
    return cypher

def _question_has_year(s: str) -> bool:
    return bool(re.search(r"\b20\d{2}\b", s or ""))

def sanitize_and_fix(cypher: str, user_question: str = "") -> str:
    lowered = " " + cypher.lower().replace("\n", " ") + " "
    if any(kw in lowered for kw in [" create "," merge "," delete "," remove "," set ",
                                    " call dbms", " apoc.trigger", " apoc.schema", " load csv",
                                    " create constraint", " drop constraint", " create index", " drop index"]):
        raise HTTPException(status_code=400, detail="Nur Leseabfragen sind erlaubt.")
    fixed = cypher.strip()
    fixed = remove_generic_labels(fixed)
    # Arrays sicher indexieren
    for prop in ["KennzahlWert", "anteilProzent"]:
        pattern = re.compile(r"(\b[A-Za-z_][A-Za-z0-9_]*)\." + re.escape(prop) + r"(?!\s*\[)")
        fixed = pattern.sub(r"coalesce(\1." + prop + r"[0], \1." + prop + r")", fixed)
    if not is_company_question(user_question):
        fixed = expand_metric_labels(fixed)
    fixed = expand_holdings(fixed)
    # Firmen-Relation robust
    fixed = re.sub(r"\[\s*:\s*hatfinanzkennzahl\s*\]",
                   "[:beziehtSichAufUnternehmen|hatFinanzkennzahl]", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"(\[\s*:[^\]]*?)\bhatfinanzkennzahl\b",
                   r"\1beziehtSichAufUnternehmen|hatFinanzkennzahl", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"(\[\s*\w+\s*:\s*)([^]\|]*?)\bhatfinanzkennzahl\b",
                   r"\1beziehtSichAufUnternehmen|hatFinanzkennzahl", fixed, flags=re.IGNORECASE)
    # Segment-Kante ungerichtet
    fixed = re.sub(r"-\s*\[\s*:\s*segmentiertNachGeschaeftsbereich\s*\]\s*->",
                   "-[:segmentiertNachGeschaeftsbereich]-", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"<-\s*\[\s*:\s*segmentiertNachGeschaeftsbereich\s*\]\s*-",
                   "-[:segmentiertNachGeschaeftsbereich]-", fixed, flags=re.IGNORECASE)
    # Umsatz nie als Tail → CONTAINS
    fixed = re.sub(
        r"toLower\(\s*([A-Za-z_]\w*)\.uri\s*\)\s*ENDS\s*WITH\s*toLower\('\/?umsatzerl(ö|oe)se'\)",
        r"( toLower(\1.uri) CONTAINS 'umsatzerlöse' OR toLower(\1.uri) CONTAINS 'umsatz' )",
        fixed, flags=re.IGNORECASE,
    )
    fixed = expand_uri_endswiths(fixed)
    fixed = expand_year_filters(fixed)
    # Rückgabespalten harmonisieren
    fixed = re.sub(r"\bAS\s+anteil\b", "AS wert", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"\bAS\s+(value|betrag|amount)\b", "AS wert", fixed, flags=re.IGNORECASE)
    fixed = re.sub(r"\bAS\s+(kennzahl|id|node|knoten)\b", "AS uri", fixed, flags=re.IGNORECASE)
    # Beziehungstyp-Alternativen: nach dem ersten Typ KEIN weiterer Doppelpunkt
    fixed = re.sub(r"\|\s*:", "|", fixed)
    # Firmenbeziehung ungerichtet
    fixed = re.sub(
        r"-\s*\[\s*:\s*(?:beziehtSichAufUnternehmen|hatFinanzkennzahl)(?:\s*\|\s*(?:beziehtSichAufUnternehmen|hatFinanzkennzahl))*\s*\]\s*(?:->|<-)",
        "-[:beziehtSichAufUnternehmen|hatFinanzkennzahl]-",
        fixed,
        flags=re.IGNORECASE
    )
    # NEW: Case-insensitive URI-Matching
    fixed = re.sub(r"uri ENDS WITH '([^']+)'", lambda m: "toLower(uri) ENDS WITH toLower('" + m.group(1) + "')", fixed)
    # NEW: Jahr-Filter entfernen, wenn die Frage kein Jahr nennt
    if not _question_has_year(user_question):
        fixed = re.sub(r"\s+AND\s+\(*\s*p\.uri\s+ENDS\s+WITH\s*['\"]#?20\d{2}['\"]\s*\)*", "", fixed, flags=re.IGNORECASE)
        fixed = re.sub(r"\s+NULLS\s+LAST\b", "", fixed, flags=re.IGNORECASE)
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
    return s.lower().replace("ä","ae").replace("ö","oe").replace("ü","ue").replace("ß","ss").strip()

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
            if "umsatzerlöse" in _norm(key):
                return {"keyword": "umsatzerlöse", "year": year}
            return {"tail": tail, "year": year}
    return None

# ===== Segment-Erkennung (ohne Jahr) =====
DI_URI = "http://www.semanticweb.org/panthers/ontologies/2025/1-Entwurf/Digital_Industries"
SI_URI = "http://www.semanticweb.org/panthers/ontologies/2025/1-Entwurf/Smart_Infrastructure"
MO_URI = "http://www.semanticweb.org/panthers/ontologies/2025/1-Entwurf/Mobility"
SH_URI = "http://www.semanticweb.org/panthers/ontologies/2025/1-Entwurf/Siemens_Healthineers"
SEG_PATTERNS = [
    (r"\bdigital\s+industries\b", DI_URI),
    (r"\bdi\b", DI_URI),
    (r"\bsmart\s+infrastructure\b", SI_URI),
    (r"\bmobility\b", MO_URI),
    (r"\bsiemens\s+healthineers\b", SH_URI),
    (r"\bhealthineers\b", SH_URI),
]

def _parse_metric_segment(q: str) -> Optional[Dict[str, Any]]:
    qn = re.sub(r"[^a-z0-9äöüß]+", " ", (q or "").lower())
    if any(w in qn for w in ["umsatz", "umsatzerlöse", "umsatzerloese"]):
        for pat, uri in SEG_PATTERNS:
            if re.search(pat, " " + qn + " "):
                return {"metric": "umsatzerlöse", "segment_uri": uri}
    return None

# ================= RAG / PDF =================
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
    if not url:
        raise ValueError("URL fehlt")
    fd, tmp_path = tempfile.mkstemp(prefix="ragpdf_", suffix=".pdf")
    os.close(fd)
    urllib.request.urlretrieve(url, tmp_path)
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        raise RuntimeError("Download fehlgeschlagen oder Datei leer")
    return tmp_path

def _query_rag(question: str, top_k: int = 6) -> List[Dict[str, Any]]:
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
    # VERBESSERUNG: Für narrative Fragen (kein Max, kein Umsatz/AE) -> keine Umwertung
    if not (needs_max or want_rev or want_oi):
        return ctx
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
        if AUTO_PDF_PATH and os.path.exists(AUTO_PDF_PATH) and count_before == 0:
            n = _ingest_pdf(AUTO_PDF_PATH)
            print(f"[RAG] Lokale PDF indiziert: {AUTO_PDF_PATH} (Chunks: {n})")
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
      [r IN rels | {{source: startNode(r).uri, target: endNode(r).uri, type: type(r)}}] AS edges
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

    # NEW: Intent-Erkennung
    fact_keywords = ["wie hoch", "was ist der", "beteiligungsquote", "ergebnismarge", "roce", "umsatzerlöse", "auftragseingang"]
    narrative_keywords = ["zukünftige", "entwicklung", "warum", "erkläre", "trends", "beeinflusst"]
    is_factual = any(k in question.lower() for k in fact_keywords)
    is_narrative = any(k in question.lower() for k in narrative_keywords) or is_outlook_question(question)

    if is_factual and not is_narrative:
        force_graph = True
    elif is_narrative:
        force_pdf = True

    print(f"Query: {question}, Mode: {'Graph' if force_graph else 'RAG' if force_pdf else 'Hybrid'}")

    # NEW: Früher Pfad für narrative Fragen (Ausblick/Prognose) -> direkt RAG
    if not force_graph and is_outlook_question(question):
        boosted_q = question + " Ausblick Prognose Outlook Guidance Erwartung Trend Digital Industries"
        ctx = _query_rag(boosted_q, top_k=12)
        ql = (question or "").lower()

        if "digital" in ql and "industr" in ql:
            filt = [c for c in (ctx or []) if "digital" in (c.get("text", "").lower()) or "industr" in (c.get("text", "").lower())]
            if filt:
                ctx = filt

        if ctx:
            snippets = []
            by_page = {}
            for c in ctx:
                meta = c.get("meta") or {}
                page = meta.get("page")
                if page:
                    by_page.setdefault(page, []).append((c.get("text") or "").strip())
            for p in sorted(by_page.keys())[:5]:
                joined = " ".join(by_page[p])
                snippets.append("— Seite " + str(p) + " —\n" + re.sub(r"\s+", " ", joined)[:1200])

            # --- Assistent-Identität & Regeln ---
            ASSISTANT_IDENTITY = (
                "Du bist ein sehr präziser, faktenbasierter Assistent für Finanz- und Geschäftsberichte "
                "(RAG über Unternehmens-PDFs + Graph-Abfragen). "
                "Du antwortest knapp, sachlich und ausschließlich mit Informationen, "
                "die im Kontext explizit belegt sind. Keine Spekulation."
            )
            OUTLOOK_RULES = (
                "Antworte nur anhand des Kontexts. "
                "Fasse die Aussagen zum erwarteten Markt-/Geschäftsverlauf (Ausblick) knapp zusammen, "
                "gerne in 2–5 Bulletpoints. Keine Spekulation, nur Inhalte aus dem Kontext."
            )

            # --- Backslash-sichere Prompt-Konstruktion ---
            joiner = "\n\n"
            context_block = joiner.join(snippets)
            question_str = str(question or "").strip()

            prompt = (
                ASSISTANT_IDENTITY + "\n\n"
                + OUTLOOK_RULES + "\n\n"
                + "Kontext:\n" + context_block + "\n\n"
                + "Frage: " + question_str + "\n\n"
                + "Antwort:"
            )

            try:
                raw = llm.invoke(prompt)
                ans = (raw.content if hasattr(raw, "content") else str(raw)).strip() or "Keine Daten gefunden."
            except Exception:
                ans = "Keine Daten gefunden."

            sources = {(c.get("meta") or {}).get("source") for c in ctx if (c.get("meta") or {}).get("source")}
            src_txt = ", ".join(sorted(s for s in sources if s)) or "PDF"

            return {
                "mode": "answer",
                "answer": ans + "\n\nQuelle: " + src_txt,
                "pdf_pages": [],
                "pdf_source": src_txt
            }

    # ---- deterministischer Kennzahl+Jahr Pfad (Graph) – vor LLM
    if not force_pdf:
        my = parse_metric_year_question(question)
        if my:
            # Keyword-Fall für Umsatzerlöse → alle Varianten (GB/Reg/EU-Tax) fürs Jahr
            if "keyword" in my:
                year = my["year"]
                params = {
                    "siemens": "http://www.semanticweb.org/panthers/ontologies/2025/1-Entwurf/Siemens_AG",
                    "metricKey": "umsatzerlöse",
                    "y1": f"/{year}",
                    "y2": f"#{year}",
                }
                cypher_exec = """
                MATCH (k)-[:beziehtSichAufUnternehmen|hatFinanzkennzahl]-(:Konzernmutter {uri:$siemens})
                MATCH (k)-[:beziehtSichAufPeriode]->(p:Geschaeftsjahr)
                WHERE (p.uri ENDS WITH $y1 OR p.uri ENDS WITH $y2)
                  AND k.KennzahlWert IS NOT NULL AND k.KennzahlWert <> []
                  AND toLower(k.uri) CONTAINS $metricKey
                OPTIONAL MATCH (k)-[:segmentiertNachGeschaeftsbereich]->(gb:Geschaeftsbereiche)
                OPTIONAL MATCH (k)-[:ausgedruecktInEinheit]->(e)
                WITH k,p,gb,e,
                CASE
                  WHEN gb IS NOT NULL THEN 'GB'
                  WHEN any(x IN k.hatKategorie WHERE toLower(x) CONTAINS 'umsatzerlöse nach regionen') THEN 'REG'
                  WHEN any(x IN k.hatKategorie WHERE toLower(x) CONTAINS 'eu-taxonomie') THEN 'EU-TAX'
                  ELSE 'TOTAL'
                END AS cat
                RETURN cat,
                       coalesce(gb.uri, '/Total') AS gruppe,
                       k.uri AS uri,
                       coalesce(k.KennzahlWert[0],k.KennzahlWert) AS wert,
                       coalesce(e.uri,'/EUR') AS einheit,
                       p.uri AS periode
                ORDER BY cat, gruppe, uri
                """
                try:
                    rows = graph.query(cypher_exec, params=params)
                    print(f"Generated Cypher: {cypher_exec}")
                except Exception as e:
                    print(f"Cypher Error: {str(e)}")
                    rows = []
                if rows:
                    total_row = next((r for r in rows if r["cat"] == "TOTAL" and r.get("wert") is not None), None)
                    if not total_row:
                        reg_vals = [r["wert"] for r in rows if r["cat"] == "REG" and r.get("wert") is not None]
                        if reg_vals:
                            total_row = {"wert": sum(reg_vals), "einheit": "/EUR", "uri": "SUM(Regionen)", "periode": f".../{year}"}
                    year_txt = str(year)
                    parts = []
                    if total_row:
                        parts.append(f"**Umsatzerlöse {year_txt} (konzernweit):** {de_format_number(total_row['wert'])} {short_name(total_row.get('einheit')) or ''}")
                    gb = [r for r in rows if r["cat"] == "GB"]
                    reg = [r for r in rows if r["cat"] == "REG"]
                    eu = [r for r in rows if r["cat"] == "EU-TAX"]
                    if gb:
                        parts.append("**Geschäftsbereiche:**\n" + "\n".join(f"- {short_name(r['gruppe'])}: {de_format_number(r['wert'])}" for r in gb))
                    if reg:
                        parts.append("**Regionen:**\n" + "\n".join(f"- {short_name(r['uri'])}: {de_format_number(r['wert'])}" for r in reg))
                    if eu:
                        parts.append("**EU-Taxonomie:**\n" + "\n".join(f"- {short_name(r['uri'])}: {r['wert']} %" for r in eu))
                    return {
                        "mode": "answer",
                        "cypher": "(deterministisch: alle Varianten)",
                        "cypher_executed": cypher_exec,
                        "rows": rows,
                        "answer": "\n\n".join(parts) if parts else "Keine Daten gefunden."
                    }
            # Tail-Fall (z. B. Auftragseingang)
            if "tail" in my:
                tail = my["tail"]
                year = my["year"]
                cypher_exec = f"""
                MATCH (k)-[:beziehtSichAufPeriode]->(p:Geschaeftsjahr)
                OPTIONAL MATCH (k)-[:beziehtSichAufUnternehmen|hatFinanzkennzahl]-(u:Konzernmutter)
                WHERE (toLower(k.uri) ENDS WITH toLower('{tail}')
                       OR toLower(replace(replace(k.uri,'-','_'),'.','_')) CONTAINS toLower('{tail}'))
                  AND (p.uri ENDS WITH '#{year}' OR p.uri ENDS WITH '/{year}')
                  AND (u.uri ENDS WITH '/Siemens_AG')
                RETURN k.uri AS uri, coalesce(k.KennzahlWert[0], k.KennzahlWert) AS wert, p.uri AS periode
                ORDER BY uri
                LIMIT 1
                """
                try:
                    rows = graph.query(cypher_exec)
                    print(f"Generated Cypher: {cypher_exec}")
                except Exception as e:
                    print(f"Cypher Error: {str(e)}")
                    rows = []
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
    # ---- deterministischer Pfad „Umsatz + Geschäftsbereich (ohne Jahr)“
    if not force_pdf:
        ms = _parse_metric_segment(question)
        if ms:
            params = {
                "siemens": "http://www.semanticweb.org/panthers/ontologies/2025/1-Entwurf/Siemens_AG",
                "seg": ms["segment_uri"],
            }
            cypher_exec = """
            WITH $seg AS segUri
            WITH segUri, toLower("_" + split(segUri,"/")[size(split(segUri,"/"))-1]) AS seg_tail
            MATCH (k)
            WHERE (toLower(k.uri) CONTAINS 'umsatzerlöse' OR toLower(k.uri) CONTAINS 'umsatz')
            OPTIONAL MATCH (k)-[:segmentiertNachGeschaeftsbereich]-(gb:Geschaeftsbereiche)
            WITH k, seg_tail, segUri, gb
            WHERE (gb.uri = segUri) OR toLower(k.uri) CONTAINS seg_tail
            MATCH (k)-[:beziehtSichAufUnternehmen|hatFinanzkennzahl]-(:Konzernmutter {uri:$siemens})
            OPTIONAL MATCH (k)-[:beziehtSichAufPeriode]->(p:Geschaeftsjahr)
            WITH k, p,
                 coalesce(k.KennzahlWert[0], k.KennzahlWert) AS wert,
                 CASE WHEN p IS NULL THEN NULL ELSE toInteger(right(p.uri,4)) END AS jahr
            WHERE jahr IS NOT NULL
            RETURN k.uri AS uri, wert, p.uri AS periode, jahr
            ORDER BY jahr DESC
            LIMIT 2
            """
            try:
                rows = graph.query(cypher_exec, params=params)
                print(f"Generated Cypher: {cypher_exec}")
            except Exception as e:
                print(f"Cypher Error: {str(e)}")
                rows = []
            if rows:
                lines = [f"- {r['jahr']}: {de_format_number(r.get('wert'))} EUR" for r in rows if r.get("jahr")]
                seg_txt = short_name(params['seg'])
                return {
                    "mode": "answer",
                    "cypher": "(deterministisch: Umsatz je Geschäftsbereich, jüngste 2 Perioden)",
                    "cypher_executed": cypher_exec,
                    "rows": rows,
                    "answer": f"Umsatzerlöse ({seg_txt}), jüngste Jahre:\n" + "\n".join(lines)
                }
                
        # ---- Fallback: Umsatz ohne Jahr – jüngste 2 Jahre
        if not rows and _want_revenue(question) and not force_pdf:
            params = {
                "siemens": "http://www.semanticweb.org/panthers/ontologies/2025/1-Entwurf/Siemens_AG"
            }
            cypher_exec = """
            MATCH (k)-[:beziehtSichAufUnternehmen|hatFinanzkennzahl]-(:Konzernmutter {uri:$siemens})
            MATCH (k)-[:beziehtSichAufPeriode]->(p:Geschaeftsjahr)
            WHERE k.KennzahlWert IS NOT NULL AND k.KennzahlWert <> []
              AND (toLower(k.uri) CONTAINS 'umsatzerlöse' OR toLower(k.uri) CONTAINS 'umsatz')
            OPTIONAL MATCH (k)-[:segmentiertNachGeschaeftsbereich]->(gb:Geschaeftsbereiche)
            OPTIONAL MATCH (k)-[:ausgedruecktInEinheit]->(e:Einheit)
            WITH k,p,gb,e, coalesce(k.KennzahlWert[0],k.KennzahlWert) AS wert, toInteger(right(p.uri,4)) AS jahr
            WHERE jahr IS NOT NULL
            RETURN jahr, k.uri AS uri, wert, coalesce(gb.uri,'/Total') AS gruppe, coalesce(e.uri,'/EUR') AS einheit
            ORDER BY jahr DESC, gruppe ASC
            LIMIT 100
            """
            try:
                rows = graph.query(cypher_exec, params=params)
                print(f"Generated Cypher: {cypher_exec}")
            except Exception as e:
                print(f"Cypher Error: {str(e)}")
                rows = []

            if rows:
                # Nimm bewusst mehrere Varianten (z. B. letzte 2 Jahre),
                # damit len(rows) > 1 wird und die Clarify-Liste greift.
                years = sorted({r["jahr"] for r in rows if r.get("jahr")}, reverse=True)[:2]
                sel = [r for r in rows if (not years) or (r.get("jahr") in years)]

                # Duplikate nach URI entfernen
                seen = set()
                deduped = []
                for r in sel:
                    uri = r.get("uri")
                    if not uri or uri in seen:
                        continue
                    seen.add(uri)
                    deduped.append({"uri": uri, "wert": r.get("wert")})

                # Übergib die Kandidaten an die Disambiguation weiter unten.
                rows = deduped
                # WICHTIG: Kein return hier – der Clarify-Block übernimmt später.

        # ---- Generischer Fallback: beliebige Kennzahl ohne Jahr -> mehrere Varianten an Clarify übergeben
        if not rows and not force_pdf:
            def _extract_alias_key(q: str) -> Optional[str]:
                ql = (q or "").lower()
                for human, tail in METRIC_ALIASES.items():
                    if human in ql:
                        # Umsatz hat bereits eigenen Fallback oben -> hier überspringen
                        if human in ("umsatzerlöse", "umsatz", "umsatzerlose"):
                            return None
                        return str(tail).lower().strip("/")
                return None

            alias = _extract_alias_key(question)
            if alias:
                params = {
                    "siemens": "http://www.semanticweb.org/panthers/ontologies/2025/1-Entwurf/Siemens_AG",
                    "alias": alias,
                }
                cypher_exec = """
                MATCH (k)-[:beziehtSichAufUnternehmen|hatFinanzkennzahl]-(:Konzernmutter {uri:$siemens})
                MATCH (k)-[:beziehtSichAufPeriode]->(p:Geschaeftsjahr)
                WHERE k.KennzahlWert IS NOT NULL AND k.KennzahlWert <> []
                  AND toLower(k.uri) CONTAINS $alias
                OPTIONAL MATCH (k)-[:ausgedruecktInEinheit]->(e:Einheit)
                WITH k, p, coalesce(k.KennzahlWert[0], k.KennzahlWert) AS wert,
                     toInteger(right(p.uri,4)) AS jahr, coalesce(e.uri,'/EUR') AS einheit
                WHERE jahr IS NOT NULL
                RETURN jahr, k.uri AS uri, wert, einheit
                ORDER BY jahr DESC, uri ASC
                LIMIT 100
                """
                try:
                    rows = graph.query(cypher_exec, params=params)
                    print(f"Generated Cypher (generic metric fallback): {cypher_exec}")
                except Exception as e:
                    print(f"Cypher Error: {str(e)}")
                    rows = []

                if rows:
                    # mehrere letzte Jahre auswählen (z. B. 2)
                    years = sorted({r["jahr"] for r in rows if r.get("jahr")}, reverse=True)[:2]
                    sel = [r for r in rows if (not years) or (r.get("jahr") in years)]

                    # Duplikate nach URI entfernen -> Kandidatenliste für Clarify
                    seen = set()
                    deduped = []
                    for r in sel:
                        uri = r.get("uri")
                        if not uri or uri in seen:
                            continue
                        seen.add(uri)
                        deduped.append({"uri": uri, "wert": r.get("wert")})

                    rows = deduped
                    # Kein return – der Clarify-Block übernimmt später.

    # ---- Graph (LLM-Cypher) – nur wenn bisher keine rows
    if not force_pdf and not rows:
        cypher_in = cypher_prompt.format(
            history=history_text or "(kein Verlauf)",
            schema=schema_text,
            labels_allow=", ".join(LABELS_ALLOW),
            rels_allow=", ".join(RELS_ALLOW),
            fewshots=FEWSHOTS,
            question=question
        )
        try:
            cypher_raw = llm.invoke(cypher_in).content.strip().strip("`")
        except Exception as e:
            print(f"LLM Error: {str(e)}")
            cypher_raw = ""
        if not cypher_raw:
            print("Graph: Keine Cypher-Query generiert")
            should_use_rag = True
        else:
            cypher_exec = sanitize_and_fix(cypher_raw, question)
            try:
                rows = graph.query(cypher_exec)
                print(f"Generated Cypher: {cypher_exec}")
            except Exception as e:
                print(f"Cypher Error: {str(e)}")
                rows = []
            # NEW: Rückfall-Brücke für narrative Fragen
            if rows and is_outlook_question(question):
                print("Graph: Irrelevante Treffer für narrative Frage – Fallback zu RAG")
                should_use_rag = True
            else:
                should_use_rag = force_pdf or (not rows and not force_graph and not is_company_question(question))
    else:
        # Falls wir den LLM-Block überspringen (z. B. weil rows schon da sind)
        should_use_rag = force_pdf

    # ---- Disambiguation & formatierte Antworten
    if not force_pdf and not should_use_rag:
        # Intent-basierte Nachfilterung – verhindert fachfremde Kennzahlen in den Kandidaten
        if rows:
            def _uri_lower(r):
                return (r.get("uri") or "").lower()
            if _want_revenue(question):
                rows = [r for r in rows if ("umsatz" in _uri_lower(r)) or ("umsatzerl" in _uri_lower(r))]
            if _want_order_intake(question):
                rows = [r for r in rows if "auftrag" in _uri_lower(r)]

        if len(rows) > 1 or (not rows and is_company_question(question)):
            if is_company_question(question):
                cq = """
                MATCH (n)
                WHERE ('Konzernmutter' IN labels(n)) OR ANY(l IN labels(n) WHERE l IN $holding)
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
        if len(rows) > 1 and not is_company_question(question)):
            opts = []
            for r in rows[:15]:  # VERBESSERUNG: Limit auf 15 erhöht
                uri = r.get("uri")
                if uri:
                    opts.append({"uri": uri, "label": prettify_tail(uri), "wert": r.get("wert")})
            if len(rows) > 15:
                opts.append({"uri": "", "label": "Mehr Optionen laden...", "wert": None})
            if opts:
                return {
                    "mode": "clarify",
                    "question": "Ich habe mehrere passende Kennzahlen gefunden. Welche meinst du genau?",
                    "options": opts,
                    "cypher_tried": cypher_exec
                }
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
            if "wert" in r0 and r0["wert"] is not None:
                return {
                    "mode": "answer",
                    "cypher": cypher_raw,
                    "cypher_executed": cypher_exec,
                    "rows": rows,
                    "answer": f"Ergebnis: {de_format_number(r0['wert'])}"
                }

    # ---- RAG (PDF)
    if should_use_rag:
        print("Graph: Keine Treffer – Fallback zu RAG")
        ctx = _query_rag(question, top_k=8)
        if ctx:
            ctx = _rerank_for_tables(question, ctx)
            needs_max = _is_superlative_question(question)
            want_rev = _want_revenue(question)
            want_oi = _want_order_intake(question)
            if needs_max and (want_rev or want_oi):
                collected: Dict[str, float] = {}
                used_pages, sources = set(), set()
                for c in ctx:
                    txt = (c.get("text") or "").strip()
                    meta = (c.get("meta") or {})
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
                sources = { (c.get("meta") or {}).get("source") for c in ctx if (c.get("meta") or {}).get("source") }
                source_name = ", ".join(sorted(sources)) if sources else None
                return {
                    "mode": "answer",
                    "answer": "Keine Daten gefunden." + (f"\n\nQuelle: {source_name}" if source_name else ""),
                    "pdf_pages": [],
                    "pdf_source": source_name or "PDF"
                }
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
    return {"mode": "answer", "answer": "Keine Daten gefunden."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
