# Advanced RAG hands-on challenge

Question answering over three reports, each a different shape of document,
each needing a different fix on top of a baseline RAG pipeline:

| Task | Document | Its problem | What fixes it |
|---|---|---|---|
| 1 | GPO 2018 Annual Report (82 pp.) | Plain prose on two-column pages | Column-aware reading order, sentence chunks, hybrid search, reranking |
| 2 | World Bank *Global Economic Prospects*, June 2026 (200 pp.) | Six regions all talk about "growth forecasts" | Headings recovered from fonts, heading path on every chunk, metadata filters, parent sections |
| 3 | Apple iMac Product Environmental Report (12 pp.) | Facts sit in tables, a chart and endnotes | Tables rebuilt from header bars, footnotes linked to claims, a vision model describes the chart |

Everything is open source and runs locally on a CPU: PyMuPDF parses, Qdrant
(local mode, a folder on disk) stores and searches, fastembed runs the BGE
embeddings, BM25 and the MiniLM cross-encoder as ONNX. The only hosted calls
are to Gemini: one per question to write the answer, and one per iMac page,
once, to describe its charts (cached in `advanced_rag/.cache/`, committed).

## Run it

Python 3.10+, no GPU needed. Only `GEMINI_API_KEY` is required.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                        # then set GEMINI_API_KEY

# The PDFs come from the challenge zip and are not committed:
unzip -j "Advanced RAG.zip" '*.pdf' -d advanced_rag/docs/

python -m advanced_rag ingest all           # ~4 min on CPU, builds naive + advanced indexes
python -m advanced_rag serve                # web page at http://localhost:8001
```

From the command line instead:

```bash
python -m advanced_rag ask gep "What is the growth outlook for South Asia in this report?"
python -m advanced_rag ask imac "What was the baseline emissions figure?" --show
python -m advanced_rag ask gpo "..." --naive   # the baseline, for comparison
python -m advanced_rag eval all              # retrieval scores, no API calls
python -m advanced_rag eval all --answers    # plus answer checks; writes advanced_rag/results/scorecard.md
```

`ask` prints the answer with numbered citations, the filter it applied, and
each source's page, kind and heading path; `--show` prints the passages. The
web page does the same, with a switch for the naive baseline and the brief's
sample questions one click away.

Qdrant's local mode lets one process open the index at a time: stop the
server before running `ingest` or `eval`.

## Results

Retrieval, measured with `python -m advanced_rag eval all`. MRR is 1.0 when
the passage holding the answer always ranks first; Hit@5 is the share of
questions with it somewhere in the top 5. One technique is added per row.

| Pipeline | GPO | World Bank | iMac |
|---|---|---|---|
| naive: plain text, 1,000-char chunks, dense | 0.73 (90%) | 0.20 (36%) | 0.54 (67%) |
| parsed + structure-aware chunks, dense | 0.80 (100%) | 0.53 (79%) | 0.96 (100%) |
| + hybrid (dense + BM25, RRF) | 0.82 (100%) | 0.55 (93%) | 0.94 (100%) |
| + cross-encoder rerank | **0.95** (100%) | 0.79 (100%) | 0.89 (100%) |
| + metadata routing (region/chapter filter) | n/a | **0.83** (100%) | n/a |
| **shipped config** | **0.95** | **0.83** | **0.94** (rerank off) |

Cells are MRR (Hit@5); 10, 14 and 12 answerable questions respectively.
Answer accuracy (`eval all --answers`) writes `advanced_rag/results/scorecard.md`.

## How it works

```
 Ingest  PDF ─ parser (per document) ─ chunks + metadata ─ BGE dense + BM25 sparse ─ Qdrant
 Ask     question ─ [route: region/chapter filter] ─ dense ∥ BM25 ─ RRF fusion
                  ─ [cross-encoder rerank top 20] ─ top 5 (or their parents) ─ Gemini ─ cited answer
```

The naive baseline is the textbook pipeline, kept as a separate collection
per document: `page.get_text()`, a cut every 1,000 characters with 100 of
overlap, dense search only.

### Shared core

| File | Does |
|---|---|
| `layout.py` | Reading order: a page is cut into bands at full-width blocks, and in each band the left column is read before the right |
| `textfix.py` | Ligatures (`ﬁ`), the World Bank font's dropped "Th" ("Te" → "The", 186 times), hyphenated line breaks |
| `chunking.py` | Whole-sentence chunks with one sentence of overlap; each sentence keeps its page |
| `store.py` | Qdrant collection with a `dense` and a `bm25` (IDF) vector per chunk; the heading path is embedded in front of the text |
| `retrieve.py` | Dense or hybrid (RRF) search, optional filter, optional rerank, parent expansion |
| `generate.py` | Answers only from numbered passages, cites them, refuses with "I could not find this in the document." |
| `evaluate.py` | MRR / Hit@5 per technique, and answer checks |

### Task 1: GPO, the baseline (`parsers/gpo.py`)

- The running footer is dropped and columns are read in order.
- Photo captions are held back until the sentence they interrupt ends: one sat
  between "over 35" and "million title retrievals" across a page turn.
- A drop cap ("T" + "he U.S. Government…") is put back on its word.
- Bold short blocks become section headings; bold table rows and contents
  leaders do not.
- Chunks are ~800 characters of whole sentences within one section.

### Task 2: World Bank, structure (`parsers/gep.py`, `route.py`)

- Chapters come from the PDF outline. Headings come from fonts: body text is
  Garamond, headings are Helvetica at 16/14 pt (section), 11 pt (subsection),
  and bold-italic Garamond (topic).
- Every chunk carries `chapter`, `region` and `box` metadata and a heading path
  such as `Chapter 2: Regional Outlooks > South Asia > Outlook`. The
  passage says "SAR"; the heading says "South Asia".
- Boxed articles (`BOX 1.1 … (continued)`) are kept apart from the chapter
  text that surrounds them.
- Tables are rebuilt row by row from word positions. Figures keep their
  caption, subtitle, panel titles and notes, not their tick labels.
- Prose is indexed as ~1,000-character children, each pointing to a
  ~3,600-character parent: small chunks match precisely, and the model reads
  the parent.
- `route.py` turns a region, a region code, a country or a named chapter into
  a filter. The country map is read from the report's own regional tables,
  which is how "the region that includes Pakistan" finds Middle East, North
  Africa, Afghanistan, and Pakistan. A chapter is filtered only when the
  question names one, because a wrong filter hides the answer.

### Task 3: iMac, visual content (`parsers/imac.py`, `vision.py`)

- Tables: each sits under a dark header bar, found in the page's vector
  drawings. Rows are rebuilt from word positions, so every value is stored
  with its row and column: `Sleep: 100V = 0.56W; 115V = 0.57W; 230V = 0.59W`.
- Footnotes: superscript markers become `[^n]`, and the endnote text from
  pages 11–12 is appended to the chunk that cites it. Each endnote is also
  indexed on its own.
- One chunk per element: blocks stacked closely in one column form a call-out
  ("76%" + "recycled content in fiber packaging"); bare chart labels are
  dropped.
- Chart: Gemini describes each page's charts and call-outs once at index
  time. Text extraction has the labels "445 kg" and "–99 kg"; only the
  description says that 445 minus 99 is 346.
- The reranker is off for this document: trained on web prose, it scores
  table rows and call-outs below sentences (see the results).

## Reading the numbers honestly

- 10–14 hand-checked questions per document, written for this exercise with
  the three samples from the brief in each set. Large effects are real;
  differences under ~0.1 MRR are noise.
- Advanced rows count a hit when the passage the model reads contains the
  evidence. For the World Bank report that is the ~3,600-character parent,
  a bigger target than the naive 1,000-character chunk.
- Answers are checked for required terms, not judged by a model.
- Thresholds in the parsers (header bar darkness, font sizes, gaps) were
  tuned on these PDFs. The ideas transfer; the numbers need re-checking on a
  new document.
