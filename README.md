# Aphasia Research Search

**Ask the aphasia literature a question in plain English and get back the passages that answer it.**

A semantic search engine over **44,160 passages from 1,217 open-access research papers on
aphasia** (2004–2026), built for the AIPI 510 semantic search activity. Every passage
carries its paper's title, authors, year, section and link, so each answer can be traced and cited.

- **Live app:** https://aphasia-search.streamlit.app
- **Code and data:** https://github.com/burakdon/aphasia-search

Queries like *"do semantic cues help patients with semantic deficits more than phonemic cues?"* or
*"how long do naming therapy gains last?"* return the Results and Discussion paragraphs where
studies actually answer them, which a keyword search would bury under every paper that mentions
"cue".

## Why this dataset

Aphasia is a language disorder, most often after a stroke, that makes finding and saying words
hard. I work on [NeuroCue](https://github.com/burakdon/neurocue), a synthetic aphasia patient and
reinforcement-learning therapy trainer. Building it meant constantly checking modelling assumptions
against the literature: does spacing help naming, which cues rescue which errors, how fast gains
fade. This index is the tool I wanted then, and will use going forward.

The patient datasets used in NeuroCue are not suitable here: they are phoneme strings and accuracy
scores (nothing to search by meaning), and the PSST/AphasiaBank transcripts may not be
redistributed. Open-access papers are both meaningful text and licensed for reuse.

## The dataset

| | |
|---|---|
| Source | [Europe PMC](https://europepmc.org) search, full text from NCBI PMC |
| Selection | "aphasia", "aphasic" or "anomia" in the title; open access; full text available |
| Licence | **CC BY or CC0 only** (checked per paper), so excerpts may be redistributed with attribution |
| Papers | 1,217 (2004–2026) |
| Passages (chunks) | 44,160, median 126 words |
| Files | `data/papers.jsonl` (one row per paper), `data/chunks.jsonl` (one row per passage), `chromadb/` |

Each chunk row in `data/chunks.jsonl` holds `id, pmcid, title, section, section_kind, year, words,
text`; `data/papers.jsonl` holds each paper's `pmcid, pmid, doi, title, authors, journal, year,
license, url, n_chunks`. Joined on `pmcid`, the two files are a reusable dataset on their own (for
RAG, classification, or a different embedding model) without Chroma.

## Chunking strategy

Scientific papers are already divided by their authors into sections and paragraphs, and a
paragraph is usually one claim with its support. So the chunker follows the paper's structure
rather than cutting fixed character windows:

1. **Parse the JATS XML** into the abstract and the body's section tree (e.g.
   `Methods > Experimental task design > Task 1`). References, tables, figures, formulas and
   citation markers are removed, as are administrative sections (funding, ethics, author
   contributions, conflicts of interest, data availability) that would only add noise.
2. **One paragraph = one chunk** by default.
3. **Merge short paragraphs** (under 60 words) with their neighbours, *within the same
   section only*, up to 220 words, so one-line lead-ins don't become context-free chunks.
4. **Split long paragraphs** on sentence boundaries into windows of about 220 words,
   repeating the last sentence of each window at the start of the next (one-sentence overlap), so no
   claim loses its lead-in.
5. **Embed with context**: the text that is embedded is `title | section path | passage`, so a
   Results paragraph about "the treated items" still knows which study and which treatment it
   belongs to. The stored document (what the app shows) is the passage alone.

220 words of scientific English plus the title and section prefix stays inside the
embedding model's 512-token window. Each chunk also gets a coarse `section_kind` (Abstract,
Introduction, Methods, Results, Discussion, Conclusion, Other) so searches can be restricted to,
say, Results only.

## Embeddings and vector store

- **Model:** [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5), 384
  dimensions, 512-token input; the same size as the starter notebook's MiniLM but stronger on
  retrieval benchmarks and with room for whole paragraphs. Queries get bge's retrieval instruction
  prefix; passages do not. Embeddings are L2-normalised.
- **Store:** a persistent [Chroma](https://www.trychroma.com) collection, `aphasia_papers`, cosine
  distance. Chroma holds the vectors and the fields the app filters on (`pmcid`, `section_kind`,
  `year`); the app joins the passage text and citation from the two dataset files by chunk id.
  Storing the text in Chroma as well tripled it (Chroma's local backend keeps each document as
  metadata plus a full-text index) and took the database from 109 MB to 470 MB, past GitHub's
  file-size limit, for no gain in a vector search.

## The app

`app.py` (Streamlit) embeds the query, searches Chroma, and shows each passage as a card with the
paper title (linked), authors, journal, year, section and a similarity score. Filters: number of
results, earliest publication year, which sections to search, and at most one passage per paper.

## Hosting

The app runs on [Streamlit Community Cloud](https://streamlit.io/cloud), deployed straight from this
repository (main file `app.py`, Python 3.11). Hugging Face now requires a PRO subscription for
Gradio and Streamlit Spaces, even on free hardware. `requirements.txt` installs the CPU-only torch
build on Linux, because the default wheel pulls in gigabytes of CUDA libraries the app never uses.

## Run it yourself

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py          # uses the committed chromadb/
python build_index.py         # rebuilds everything from scratch (about 10 minutes)
```

```
aphasia-search/
├── app.py               # the search app (Streamlit)
├── build_index.py       # fetch → parse → chunk → embed → Chroma
├── chromadb/            # the vector database (chroma.sqlite3 + index)
├── data/
│   ├── papers.jsonl     # paper metadata
│   └── chunks.jsonl     # every passage with its citation metadata
├── requirements.txt
└── README.md
```

## Attribution and limits

All text excerpts come from papers published under CC BY (or CC0); each result names and links its
source, and reuse should cite the original paper. The index covers only open-access papers with
CC BY/CC0 licences, so it is a sample of the literature, not all of it: paywalled and
non-commercial-licensed work is missing. Section labels come from the authors' own headings and
are occasionally wrong. This is a research and course tool, not medical advice.
