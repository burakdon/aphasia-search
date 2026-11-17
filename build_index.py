"""Build the aphasia research vector database from scratch.

    python build_index.py              # fetch, parse, chunk, embed (about 10-20 minutes the first time)
    python build_index.py --skip-fetch # re-chunk and re-embed from the cached full texts

Pipeline:
  1. Search Europe PMC for open-access papers with "aphasia", "aphasic" or "anomia" in the title that
     have full text and a CC BY or CC0 licence (so the text may be redistributed with attribution).
  2. Download each paper's JATS full-text XML from NCBI PMC, 50 per request (cached under
     data/xml/, not committed).
  3. Parse the abstract and body into sections, dropping references, tables, figures and the
     administrative sections (funding, ethics, author contributions, conflicts, data availability).
  4. Chunk (see chunk_section): paragraphs are the unit; small neighbouring paragraphs in the same
     section are merged, long ones are split on sentence boundaries with one sentence of overlap.
  5. Embed "title | section | text" with BAAI/bge-small-en-v1.5 and store in a persistent Chroma
     collection (cosine distance) under chromadb/, with citation metadata on every chunk.

Outputs: data/papers.jsonl (one row per paper), data/chunks.jsonl (one row per chunk, the reusable
dataset), chromadb/ (the vector database the app loads).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
XML_DIR = DATA / "xml"
DB_DIR = ROOT / "chromadb"
COLLECTION = "aphasia_papers"
MODEL = "BAAI/bge-small-en-v1.5"

API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
QUERY = ('(TITLE:aphasia OR TITLE:aphasic OR TITLE:anomia) AND OPEN_ACCESS:y AND HAS_FT:y '
         'AND (LICENSE:"cc by" OR LICENSE:"cc0")')
ALLOWED_LICENCES = {"cc by", "cc0"}  # exact values; the search can also match cc by-nc etc.
HEADERS = {"User-Agent": "aphasia-search/0.1 (course project; semantic search over open-access papers)"}

# Chunk sizes in words. bge-small reads up to 512 tokens; ~220 words of scientific English plus the
# title/section prefix stays well inside that, while keeping each chunk about one idea.
TARGET_WORDS = 220
MIN_WORDS = 60

SKIP_SEC_TYPES = {"data-availability", "ethics-statement", "author-contributions", "coi-statement",
                  "funding-information", "ai-statement", "disclaimer", "supplementary-material",
                  "acknowledgments", "abbreviations", "conflict", "supplementary-materials"}
SKIP_TITLE = re.compile(
    r"^(acknowledg|funding|financial|conflicts? of interest|competing interest|declaration|"
    r"author(s')? contribution|contributions|data (availability|sharing|access)|availability of data|"
    r"ethic|consent|supplementary|supporting information|abbreviation|publisher|disclaimer|"
    r"generative ai|open practices|orcid|references?$|notes?$|.*study group$|collaborators|investigators$|"
    r"members of|group authors?)", re.I)
DROP_TAGS = {"table-wrap", "fig", "table", "disp-formula", "inline-formula", "graphic", "media",
             "supplementary-material", "ref-list", "fn-group", "xref"}


# ---------------------------------------------------------------- 1. search

def search() -> list[dict]:
    papers, cursor = [], "*"
    while True:
        r = requests.get(f"{API}/search", headers=HEADERS, timeout=60, params={
            "query": QUERY, "format": "json", "resultType": "core", "pageSize": 1000, "cursorMark": cursor})
        r.raise_for_status(); d = r.json()
        for x in d["resultList"]["result"]:
            lic = (x.get("license") or "").lower().strip()
            if not x.get("pmcid") or lic not in ALLOWED_LICENCES:
                continue
            papers.append({
                "pmcid": x["pmcid"], "pmid": x.get("pmid"), "doi": x.get("doi"),
                "title": clean(x.get("title", "")), "authors": x.get("authorString", ""),
                "journal": (x.get("journalInfo") or {}).get("journal", {}).get("title", "") or x.get("bookOrReportDetails", {}).get("publisher", ""),
                "year": int(x["pubYear"]) if str(x.get("pubYear", "")).isdigit() else None,
                "license": lic, "url": f"https://europepmc.org/article/PMC/{x['pmcid']}",
            })
        nxt = d.get("nextCursorMark")
        if not nxt or nxt == cursor or not d["resultList"]["result"]:
            break
        cursor = nxt
    seen, out = set(), []
    for p in papers:
        if p["pmcid"] not in seen:
            seen.add(p["pmcid"]); out.append(p)
    return out


def article_pmcid(article: ET.Element) -> str | None:
    for aid in article.iter("article-id"):
        if aid.get("pub-id-type") in ("pmc", "pmcid") and aid.text:
            v = aid.text.strip()
            return v if v.startswith("PMC") else f"PMC{v}"
    return None


def fetch_xml(pmcids: list[str], batch: int = 50) -> int:
    """Download JATS full text from NCBI's PMC E-utilities, 50 articles per request (Europe PMC's
    per-article endpoint takes ~15 s each). Each article is cached as data/xml/<PMCID>.xml."""
    todo = [p for p in pmcids if not (XML_DIR / f"{p}.xml").exists()]
    for i in range(0, len(todo), batch):
        ids = ",".join(p[3:] for p in todo[i:i + batch])
        for attempt in range(4):
            try:
                r = requests.post(EFETCH, headers=HEADERS, timeout=120,
                                  data={"db": "pmc", "id": ids, "retmode": "xml", "tool": "aphasia-search"})
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(3 * (attempt + 1))
        else:
            print(f"  batch {i // batch} failed"); continue
        root = ET.fromstring(r.content)
        for art in root.findall("article"):
            pid = article_pmcid(art)
            if pid:
                (XML_DIR / f"{pid}.xml").write_bytes(ET.tostring(art, encoding="utf-8"))
        print(f"  fetched {min(i + batch, len(todo))}/{len(todo)}", flush=True)
        time.sleep(0.4)  # NCBI asks for at most 3 requests a second without an API key
    return sum((XML_DIR / f"{p}.xml").exists() for p in pmcids)


# ---------------------------------------------------------------- 2. parse

_EMPTIED = re.compile(r"[\(\[]((?:\s|[,;:–\-]|e\.g\.|i\.e\.|cf\.|see|and|also|for (?:a )?reviews?)*)[\)\]]", re.I)


def clean(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    s = _EMPTIED.sub("", s)  # brackets left empty once citation markers are dropped: "(e.g.,,)", "[,]"
    s = re.sub(r"\s+([,;:)\]])", r"\1", s)
    s = re.sub(r"\s+\.(?=\s|$)", ".", s)
    s = re.sub(r",{2,}", ",", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def text_of(el: ET.Element) -> str:
    """Text of an element without tables, figures, formulas or citation markers."""
    parts = []
    def walk(e):
        if e.tag in DROP_TAGS:
            if e.tail: parts.append(e.tail)
            return
        if e.text: parts.append(e.text)
        for c in e: walk(c)
        if e.tail and e is not el: parts.append(e.tail)
    walk(el)
    return clean("".join(parts))


def keep_section(sec: ET.Element, title: str) -> bool:
    st = (sec.get("sec-type") or "").lower()
    return not (st in SKIP_SEC_TYPES or any(k in st for k in ("contrib", "coi", "funding", "ethic", "availability"))
                or SKIP_TITLE.match(title or ""))


def sections(root: ET.Element) -> list[tuple[str, list[str]]]:
    """[(section path, [paragraph, ...]), ...] in document order: abstract first, then the body."""
    out = []
    for ab in root.iter("abstract"):
        if ab.get("abstract-type") in ("graphical", "teaser", "toc"):
            continue
        paras = [text_of(p) for p in ab.iter("p")]
        if any(paras):
            out.append(("Abstract", [p for p in paras if p]))
        break
    body = root.find(".//body")
    if body is None:
        return out

    def walk(node, path):
        loose = [text_of(p) for p in node.findall("p")]
        loose = [p for p in loose if p]
        if loose:
            out.append((" > ".join(path) or "Body", loose))
        for sec in node.findall("sec"):
            t = sec.find("title"); title = text_of(t) if t is not None else ""
            if not keep_section(sec, title):
                continue
            walk(sec, path + ([title] if title else []))
    walk(body, [])
    return out


# ---------------------------------------------------------------- 3. chunk

SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def chunk_section(paras: list[str]) -> list[str]:
    """Paragraph-based chunking.
    - A paragraph is the natural unit of one idea, so it is the default chunk.
    - Paragraphs shorter than MIN_WORDS are merged with their neighbours (same section only) until the
      chunk reaches TARGET_WORDS, so short lead-ins do not become weak, context-free chunks.
    - Paragraphs longer than TARGET_WORDS are split on sentence boundaries into ~TARGET_WORDS windows,
      repeating the last sentence of one window at the start of the next so no claim loses its lead-in.
    """
    units: list[str] = []
    for p in paras:
        words = p.split()
        if len(words) <= TARGET_WORDS:
            units.append(p); continue
        sents = SENT.split(p); cur: list[str] = []
        for s in sents:
            if cur and len(" ".join(cur + [s]).split()) > TARGET_WORDS:
                units.append(" ".join(cur)); cur = [cur[-1]]  # one sentence of overlap
            cur.append(s)
        if cur:
            units.append(" ".join(cur))
    # a paragraph with no sentence breaks (lists, rosters) can still be too long: cut it into word
    # windows with a short overlap so nothing exceeds the model's input
    sized: list[str] = []
    for u in units:
        w = u.split()
        if len(w) <= int(TARGET_WORDS * 1.5):
            sized.append(u); continue
        for i in range(0, len(w), TARGET_WORDS - 30):
            sized.append(" ".join(w[i:i + TARGET_WORDS]))
            if i + TARGET_WORDS >= len(w):
                break
    units = sized
    chunks: list[str] = []
    for u in units:
        if chunks and (len(chunks[-1].split()) < MIN_WORDS or len(u.split()) < MIN_WORDS) \
                and len((chunks[-1] + " " + u).split()) <= TARGET_WORDS:
            chunks[-1] = chunks[-1] + " " + u
        else:
            chunks.append(u)
    return [c for c in chunks if len(c.split()) >= 25]


def section_kind(path: str) -> str:
    """Coarse section label for filtering in the app."""
    head = path.split(" > ")[0].lower()
    for kind, keys in [("Abstract", ["abstract"]), ("Introduction", ["intro", "background"]),
                       ("Methods", ["method", "material", "participant", "procedure", "design", "stimul", "analys", "measure"]),
                       ("Results", ["result", "finding"]), ("Discussion", ["discussion", "limitation", "implication", "future"]),
                       ("Conclusion", ["conclu", "summary"])]:
        if any(k in head for k in keys):
            return kind
    return "Other"


def build_chunks(papers: list[dict]) -> list[dict]:
    rows = []
    for p in papers:
        path = XML_DIR / f"{p['pmcid']}.xml"
        if not path.exists():
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            continue
        n = 0
        for sec_path, paras in sections(root):
            for text in chunk_section(paras):
                rows.append({"id": f"{p['pmcid']}-{n:03d}", "pmcid": p["pmcid"], "title": p["title"],
                             "section": sec_path[:200], "section_kind": section_kind(sec_path),
                             "year": p["year"] or 0, "words": len(text.split()), "text": text})
                n += 1
        p["n_chunks"] = n
    return rows


# ---------------------------------------------------------------- 4. embed + store

def embed_and_store(rows: list[dict]) -> None:
    """Embed every chunk (re-using cached vectors for unchanged chunks) and write the Chroma database.

    Chroma holds the vectors plus the three fields the app filters on (pmcid, section_kind, year).
    The passage text and citation details live in data/chunks.jsonl and data/papers.jsonl and are
    joined by chunk id at query time: Chroma's local backend stores each document three times
    (metadata, full-text index content and index data), which took the database from about 100 MB
    to 470 MB and over GitHub's per-file limit."""
    import hashlib

    import chromadb
    import numpy as np
    from chromadb import Schema, VectorIndexConfig
    from chromadb.config import Settings

    texts = [f"{r['title']} | {r['section']} | {r['text']}" for r in rows]
    keys = [hashlib.sha1(f"{MODEL}\n{t}".encode()).hexdigest() for t in texts]
    cache_vec, cache_keys = DATA / "emb_cache.npy", DATA / "emb_cache_keys.json"
    cached: dict[str, np.ndarray] = {}
    if cache_vec.exists() and cache_keys.exists():
        cached = dict(zip(json.loads(cache_keys.read_text()), np.load(cache_vec)))
    todo = [i for i, k in enumerate(keys) if k not in cached]
    if todo:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL)
        t0 = time.time()
        new = model.encode([texts[i] for i in todo], batch_size=64, show_progress_bar=True, normalize_embeddings=True)
        for i, v in zip(todo, new):
            cached[keys[i]] = v
        print(f"embedded {len(todo)} new chunks in {time.time() - t0:.0f}s ({len(rows) - len(todo)} from cache)")
    else:
        print(f"all {len(rows)} chunk embeddings from cache")
    emb = np.stack([cached[k] for k in keys]).astype(np.float32)
    np.save(cache_vec, emb); cache_keys.write_text(json.dumps(keys))

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
    client = chromadb.PersistentClient(path=str(DB_DIR), settings=Settings(anonymized_telemetry=False))
    schema = Schema(); schema.create_index(VectorIndexConfig(space="cosine"))
    col = client.create_collection(COLLECTION, schema=schema, metadata={"embedding_model": MODEL})
    step = 4000
    for i in range(0, len(rows), step):
        b = rows[i:i + step]
        col.add(ids=[r["id"] for r in b], embeddings=emb[i:i + step].tolist(),
                metadatas=[{"pmcid": r["pmcid"], "section_kind": r["section_kind"], "year": r["year"]} for r in b])
    print(f"stored {col.count()} vectors in {DB_DIR.relative_to(ROOT)}/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-fetch", action="store_true", help="use data/papers.jsonl and the cached XML")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True); XML_DIR.mkdir(exist_ok=True)

    if args.skip_fetch:
        papers = [json.loads(l) for l in (DATA / "papers.jsonl").read_text().splitlines()]
    else:
        papers = search(); print(f"search: {len(papers)} CC BY / CC0 papers with full text")
        got = fetch_xml([p["pmcid"] for p in papers])
        print(f"full text: {got} downloaded or cached, {len(papers) - got} unavailable")

    rows = build_chunks(papers)
    papers = [p for p in papers if p.get("n_chunks")]
    papers.sort(key=lambda p: (-(p["year"] or 0), p["pmcid"]))
    (DATA / "papers.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in papers))
    (DATA / "chunks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    words = [r["words"] for r in rows]
    print(f"chunks: {len(rows)} from {len(papers)} papers; words per chunk median "
          f"{sorted(words)[len(words) // 2]}, min {min(words)}, max {max(words)}")
    embed_and_store(rows)


if __name__ == "__main__":
    main()
