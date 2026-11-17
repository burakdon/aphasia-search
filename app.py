"""Aphasia Research Search: semantic search over open-access aphasia papers (Streamlit).

Loads the Chroma database built by build_index.py (chromadb/: the vectors plus pmcid, section kind
and year for filtering) and the dataset files it was built from (data/chunks.jsonl for the passage
text, data/papers.jsonl for citations), embeds the query with the same model, and shows the
closest passages.

    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
COLLECTION = "aphasia_papers"
MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this instruction in front of short queries (not in front of the passages)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SECTION_KINDS = ["Abstract", "Introduction", "Methods", "Results", "Discussion", "Conclusion", "Other"]
EXAMPLES = [
    "Do semantic cues help patients with semantic deficits more than phonemic cues?",
    "spaced retrieval practice versus massed practice for naming",
    "how long do naming therapy gains last after treatment ends",
    "errorless versus errorful learning in anomia treatment",
    "transcranial direct current stimulation combined with speech therapy",
    "app-based aphasia therapy delivered at home",
]

st.set_page_config(page_title="Aphasia Research Search", page_icon="🧠", layout="centered")


@st.cache_resource(show_spinner="Loading the embedding model and the index…")
def load():
    import chromadb
    from chromadb.config import Settings
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL, device="cpu")
    client = chromadb.PersistentClient(path=str(ROOT / "chromadb"), settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(COLLECTION)
    papers = {}
    for line in (ROOT / "data" / "papers.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            p = json.loads(line); papers[p["pmcid"]] = p
    chunks = {}
    with open(ROOT / "data" / "chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); chunks[r["id"]] = r
    return model, collection, papers, chunks


def short_authors(a: str) -> str:
    names = [n.strip() for n in (a or "").rstrip(".").split(",") if n.strip()]
    return names[0] + " et al." if len(names) > 2 else " and ".join(names)


def section_path(c: dict) -> str:
    """The section path without a head that only repeats the tag (Results > Results > ...)."""
    parts = [x for x in c.get("section", "").split(" > ") if x]
    while parts and parts[0].strip().lower() in (c.get("section_kind", "").lower(), "abstract", "body"):
        parts = parts[1:]
    return " › ".join(parts)


def search(query: str, k: int, year_from: int, kinds: list[str], one_per_paper: bool) -> list[dict]:
    model, collection, papers, chunks = load()
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
        seen.add(meta["pmcid"])
        c = chunks[cid]
        hits.append({**papers[c["pmcid"]], **c, "score": 1 - dist})  # cosine similarity
        if len(hits) == k:
            break
    return hits


model, collection, papers, chunks = load()
years = sorted(p["year"] for p in papers.values() if p.get("year"))

st.title("Aphasia Research Search")
st.markdown(
    f"Ask a question in plain English and get the passages from the research literature that answer it, "
    f"not just keyword hits. Searches **{collection.count():,} passages from {len(papers):,} open-access "
    f"aphasia papers** ({years[0]}–{years[-1]}, CC BY licensed, via Europe PMC). Every result links to its paper.")

with st.sidebar:
    st.header("Filters")
    k = st.slider("Results", 1, 20, 6)
    year_from = st.slider("Published from", years[0], years[-1], years[0])
    kinds = st.multiselect("Sections to search", SECTION_KINDS, default=SECTION_KINDS)
    one = st.checkbox("At most one passage per paper", value=True)
    st.divider()
    st.caption("Embeddings: BAAI/bge-small-en-v1.5 · Vector store: Chroma (cosine) · "
               "[Code and dataset](https://github.com/burakdon/aphasia-search)")


def use_example(text: str) -> None:
    st.session_state.query = text


with st.form("search", border=False):
    st.text_input("Question or topic", key="query",
                  placeholder="e.g. does spacing practice help word retrieval in aphasia?")
    st.form_submit_button("Search", type="primary")

with st.expander("Example questions", expanded=not st.session_state.get("query")):
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES):
        cols[i % 2].button(ex, key=f"ex{i}", on_click=use_example, args=(ex,), use_container_width=True)

query = (st.session_state.get("query") or "").strip()
if query:
    hits = search(query, k, year_from, kinds, one)
    if not hits:
        st.info("No passages match those filters.")
    for i, h in enumerate(hits, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. [{h['title']}]({h['url']})**")
            doi = f" · [doi](https://doi.org/{h['doi']})" if h.get("doi") else ""
            st.caption(f"{short_authors(h.get('authors', ''))} · {h.get('journal', '')} · {h.get('year') or 'n.d.'}{doi}")
            st.markdown(f":blue-badge[{h['section_kind']}] {section_path(h)} &nbsp; :gray[match {h['score']:.2f}]")
            st.write(h["text"])

st.divider()
st.caption("Research prototype for a course activity. Passages are excerpts from papers under CC BY 4.0; cite the "
           "original paper, linked on each result. Not medical advice.")
