"""Aphasia Research Search: semantic search over open-access aphasia papers.

Loads the Chroma database built by build_index.py (chromadb/: the vectors plus pmcid, section kind
and year for filtering) and the dataset files it was built from (data/chunks.jsonl for the passage
text, data/papers.jsonl for citations), embeds the query with the same model, and serves a Gradio
search page. Run locally with `python app.py`; on Hugging Face Spaces this file is
the entry point.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import chromadb
import gradio as gr
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
COLLECTION = "aphasia_papers"
MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this instruction in front of short queries (not in front of the passages)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SECTION_KINDS = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion", "Other"]

model = SentenceTransformer(MODEL)
client = chromadb.PersistentClient(path=str(ROOT / "chromadb"), settings=Settings(anonymized_telemetry=False))
collection = client.get_collection(COLLECTION)

papers = [json.loads(l) for l in (ROOT / "data" / "papers.jsonl").read_text().splitlines() if l.strip()]
PAPER = {p["pmcid"]: p for p in papers}
CHUNK = {}
with open(ROOT / "data" / "chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line); CHUNK[r["id"]] = r
YEARS = sorted(p["year"] for p in papers if p.get("year"))
MIN_YEAR, MAX_YEAR = (YEARS[0], YEARS[-1]) if YEARS else (1990, 2026)


def short_authors(a: str) -> str:
    names = [n.strip() for n in a.rstrip(".").split(",") if n.strip()]
    return names[0] + " et al." if len(names) > 2 else " and ".join(names)


def section_path(meta: dict) -> str:
    """The section path without a head that only repeats the tag (Results > Results > ...)."""
    parts = [x for x in meta.get("section", "").split(" > ") if x]
    while parts and parts[0].strip().lower() in (meta.get("section_kind", "").lower(), "abstract", "body"):
        parts = parts[1:]
    return " › ".join(parts)


def card(i: int, cid: str, dist: float) -> str:
    c = CHUNK[cid]; meta = {**PAPER[c["pmcid"]], **c}; doc = c["text"]
    score = 1 - dist  # cosine similarity
    doi = f' · <a href="https://doi.org/{html.escape(meta["doi"])}" target="_blank">doi</a>' if meta.get("doi") else ""
    return f"""
<div class="hit">
  <div class="hit-top"><span class="rank">{i}</span>
    <a class="hit-title" href="{html.escape(meta['url'])}" target="_blank">{html.escape(meta['title'])}</a></div>
  <div class="hit-meta">{html.escape(short_authors(meta.get('authors', '')))} · {html.escape(meta.get('journal', ''))} · {meta.get('year') or 'n.d.'}{doi}</div>
  <div class="hit-sec"><span class="tag">{html.escape(meta.get('section_kind', ''))}</span> {html.escape(section_path(meta))}
    <span class="score" title="cosine similarity between your query and this passage">match {score:.2f}</span></div>
  <div class="hit-text">{html.escape(doc)}</div>
</div>"""


def search(query: str, k: int, year_from: int, kinds: list[str], one_per_paper: bool) -> str:
    query = (query or "").strip()
    if not query:
        return '<div class="empty">Type a question or topic above.</div>'
    k = int(k)
    conds = [{"year": {"$gte": int(year_from)}}]
    if kinds and set(kinds) != set(SECTION_KINDS):
        conds.append({"section_kind": {"$in": list(kinds)}})
    where = conds[0] if len(conds) == 1 else {"$and": conds}
    q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q, n_results=k * 5 if one_per_paper else k, where=where,
                           include=["metadatas", "distances"])
    hits, seen = [], set()
    for cid, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
        if one_per_paper and meta["pmcid"] in seen:
            continue
        seen.add(meta["pmcid"]); hits.append((cid, dist))
        if len(hits) == k:
            break
    if not hits:
        return '<div class="empty">No passages match those filters.</div>'
    return "".join(card(i + 1, *h) for i, h in enumerate(hits))


CSS = """
.hit { text-align: left; border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 14px 16px; margin: 0 0 12px; background: var(--background-fill-secondary); }
.hit-top { display: flex; gap: 10px; align-items: baseline; }
.rank { font-weight: 700; color: var(--color-accent); min-width: 1.2em; }
.hit-title { font-weight: 650; font-size: 1.04em; text-decoration: none; }
.hit-title:hover { text-decoration: underline; }
.hit-meta { color: var(--body-text-color-subdued); font-size: .9em; margin: 4px 0 6px 2.2em; }
.hit-sec { font-size: .86em; color: var(--body-text-color-subdued); margin: 0 0 8px 2.2em; }
.tag { display: inline-block; padding: 1px 8px; border-radius: 999px; background: var(--color-accent-soft); color: var(--body-text-color); font-weight: 600; margin-right: 4px; }
.score { float: right; font-variant-numeric: tabular-nums; }
.hit-text { margin-left: 2.2em; line-height: 1.55; }
.empty { color: var(--body-text-color-subdued); padding: 20px 4px; }
"""

EXAMPLES = [
    "Do semantic cues help patients with semantic deficits more than phonemic cues?",
    "spaced retrieval practice versus massed practice for naming",
    "errorless versus errorful learning in anomia treatment",
    "how long do naming therapy gains last after treatment ends",
    "transcranial direct current stimulation combined with speech therapy",
    "computer or app-based aphasia therapy delivered at home",
    "predictors of recovery from post-stroke aphasia",
    "primary progressive aphasia word retrieval treatment",
]

n_papers, n_chunks = len(papers), collection.count()
with gr.Blocks(title="Aphasia Research Search") as demo:
    gr.Markdown(f"""# Aphasia Research Search
Ask a question in plain English and get the passages from the research literature that answer it, not just keyword hits.
Searches **{n_chunks:,} passages from {n_papers:,} open-access aphasia papers** ({MIN_YEAR}–{MAX_YEAR}, CC BY licensed, via Europe PMC).
Every result links to its paper.""")
    with gr.Row():
        query = gr.Textbox(label="Question or topic", placeholder="e.g. does spacing practice help word retrieval in aphasia?", scale=5, autofocus=True)
        go = gr.Button("Search", variant="primary", scale=1)
    with gr.Accordion("Filters", open=False):
        with gr.Row():
            k = gr.Slider(1, 20, value=6, step=1, label="Results")
            year_from = gr.Slider(MIN_YEAR, MAX_YEAR, value=MIN_YEAR, step=1, label="Published from")
        kinds = gr.CheckboxGroup(SECTION_KINDS, value=SECTION_KINDS, label="Sections to search")
        one = gr.Checkbox(value=True, label="At most one passage per paper")
    out = gr.HTML('<div class="empty">Type a question or topic above, or pick an example.</div>')
    gr.Examples(EXAMPLES, inputs=query, label="Try")
    gr.Markdown("""<small>Research prototype for a course activity. Passages are excerpts from papers under CC BY 4.0; cite the original
paper, linked on each result. Not medical advice.</small>""")
    inputs = [query, k, year_from, kinds, one]
    go.click(search, inputs, out)
    query.submit(search, inputs, out)

if __name__ == "__main__":
    demo.launch(css=CSS)
