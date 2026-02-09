# KSP Development Knowledge Extraction: Chapter-Level Classification & Policy Extraction

## Project Overview

Extract and classify development knowledge from KSP (Knowledge Sharing Program) advisory reports at the chapter level using LLM + RAG. For each chapter/sub-chapter, the pipeline performs taxonomy classification, knowledge type classification, Korean policy extraction, and theory linking to development economics textbooks.

**Research Questions:**
1. Can we systematically extract prescriptive knowledge (policies, programs) from KSP development experience reports?
2. How do these extracted policies relate to propositional knowledge (development economics theory from textbooks)?
3. What alternative categorizations of development knowledge emerge beyond traditional sectoral classifications?

## Current Phase: Small-Scale Pilot

**Scope:**
- **4 KSP advisory reports** (VNM 2009, SLV 2014, KAZ 2023, QAT 2023)
- **2 development economics textbooks** (Perkins 2012, Todaro 2012)
- **Chapter-level analysis** with combined LLM prompt
- **Textbook RAG** for theory linking (ChromaDB)

**Goals:**
- Validate the chapter-level extraction approach
- Test taxonomy classification accuracy
- Optimize the combined LLM prompt
- Assess theory-practice linking quality

## Architecture (v2 -- Chapter-Level)

### Key Design Decisions

1. **No KSP RAG** -- chapters are read directly into the LLM context (chapters are 1-10 pages, within Claude's context window). No chunking, embedding, or ChromaDB collection for KSP reports.
2. **Textbook RAG kept** -- ChromaDB collection `textbooks_pilot` stores ~7K textbook chunks for theory retrieval.
3. **ChapterExtractor** -- font-size heuristics detect chapter/sub-chapter headers from PDF structure.
4. **Combined LLM prompt** -- single API call per chapter handles all 4 tasks (taxonomy + knowledge type + policy extraction + theory linking).
5. **Full taxonomy in prompt** -- all 6 sectors, ~140 keywords embedded directly (~3K tokens).
6. **"Not Applicable"** sentinel -- used when a chapter has no Korean policies or no relevant theory links.

### Pipeline Flow

```
Phase 1: Chapter Extraction
  KSP PDFs -> ChapterExtractor (font-size heuristics) -> chapters/sub-chapters

Phase 2: Textbook Indexing (run once)
  Textbook PDFs -> LangChain chunking (768 tokens) -> sentence-transformers -> ChromaDB

Phase 3: Combined Analysis (per chapter)
  For each chapter:
    1. Query textbook ChromaDB for theory context (top_k=3)
    2. Build combined prompt (taxonomy + knowledge types + chapter text + theory context)
    3. Call Claude Sonnet API -> parse JSON response
    4. Save to chapter_analysis.json

Phase 4: Visualization
  chapter_analysis.json -> sector distribution, knowledge type charts, heatmaps, network graphs
```

## Technical Stack

- **Platform:** Google Colab
- **Storage:** Google Drive (persistence across sessions)
- **PDF Processing:** PyMuPDF (fitz) -- chapter extraction via font-size heuristics
- **Textbook Chunking:** LangChain RecursiveCharacterTextSplitter (768 tokens, 50 overlap)
- **Embeddings:** sentence-transformers/all-MiniLM-L6-v2 (384-dim vectors)
- **Vector Database:** ChromaDB -- textbook collection only (`textbooks_pilot`)
- **LLM:** Anthropic Claude Sonnet 4 (`claude-sonnet-4-20250514`) -- temperature 0.1, max_tokens 4096
- **Visualization:** Matplotlib, Seaborn, NetworkX

## Notebook Structure

`ksp_pilot_complete.ipynb` -- 29 cells (17 code, 12 markdown), 10 sections:

| Section | Cells | Description |
|---------|-------|-------------|
| 1. Setup & Installation | 2-5 | pip install, Drive mount, directory creation |
| 2. Configuration | 6-8 | `Config` dataclass, API key setup |
| 3. Taxonomy Reference | 9-10 | `TAXONOMY` dict (6 sectors, ~140 keywords), `KNOWLEDGE_TYPES`, helpers |
| 4. Chapter Extraction | 11-13 | `ChapterExtractor` class, test on one report |
| 5. Textbook Vector Store | 14-17 | `VectorStore` class, textbook indexing |
| 6. Process All KSP Reports | 18-19 | `extract_all_chapters()`, save `chapter_summaries.json` |
| 7. LLM Classification + Extraction | 20-21 | `ChapterAnalyzer` class with combined prompt |
| 8. Run Analysis | 22-23 | `run_full_analysis()`, save `chapter_analysis.json` |
| 9. Visualization | 24-26 | Charts, heatmaps, per-report detailed view, network graphs |
| 10. Summary | 27-28 | Statistics and next steps |

## Key Classes

### Config (Section 2)
```python
@dataclass
class Config:
    project_dir: str = '/content/drive/MyDrive/KM4D_v0'
    textbook_chunk_size: int = 768
    chunk_overlap: int = 50
    embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2'
    llm_model: str = 'claude-sonnet-4-20250514'
    temperature: float = 0.1
    max_tokens: int = 4096
    textbook_top_k: int = 3
    textbook_collection: str = 'textbooks_pilot'
```

### ChapterExtractor (Section 4)
- `_collect_text_blocks()` -- extracts all text spans with font size and page info
- `_identify_header_thresholds()` -- body_size = most common font size; header_sizes = larger than body
- `extract_chapters()` -- returns list of dicts: chapter_title, chapter_level (1=chapter, 2=sub-chapter), content, page_start, page_end, content_length

### VectorStore (Section 5)
- Wraps sentence-transformers + ChromaDB
- `add_documents()` -- embed and store chunks
- `search()` -- query by text, returns top-k results
- Used for textbook collection only

### ChapterAnalyzer (Section 7)
- `_get_theory_context()` -- queries textbook store with chapter title + content snippet
- `_build_prompt()` -- constructs combined prompt with taxonomy, knowledge types, chapter text, theory context
- `analyze_chapter()` -- calls Claude API, parses JSON, attaches chapter metadata
- Truncates chapter content at 15,000 chars
- 1-second rate limiting between API calls

## Taxonomy

6 sectors from `Taxonomy_20250925.pdf`:
1. **(1) Economic Policy** -- Macroeconomic, Growth, Investment & Private Sector
2. **(2) Social Services** -- Education, Health, Social Protection, Gender
3. **(3) Digital Innovation** -- Digital Policy, Infrastructure, Transformation, Emerging Tech
4. **(4) Production & Trade** -- Agriculture, Industry, Trade Policy
5. **(5) Infrastructure** -- Transport, Water & Sanitation, Urban & Rural Development
6. **(6) Energy & Environment** -- Environmental Policy, Climate Change, Energy

~140 keywords at the most specific level. Full hierarchy embedded in the LLM prompt.

## Knowledge Types

1. Contextual background and situation analysis
2. Policy implementation and coordinating mechanism
3. Technical methodology and analytical framework
4. Recommendations and future directions

## Output Schema

Main output: `data/results/chapter_analysis.json`

Per-chapter entry:
```json
{
  "report_id": "report_filename_stem",
  "chapter_title": "chapter or sub-chapter title",
  "chapter_level": 1 or 2,
  "page_start": int,
  "page_end": int,
  "content_length": int,
  "taxonomy_classification": {
    "sectors": [{"sector": "...", "sub_sector_l1": "...", "sub_sector_l2": "...", "keywords": ["..."]}],
    "knowledge_type": "one of 4 types",
    "confidence": "high|medium|low",
    "reasoning": "brief explanation"
  },
  "korean_policies": [
    {
      "policy_name": "string",
      "year_initiated": int or null,
      "organization": "string or null",
      "challenge_addressed": "string",
      "policy_instruments": ["string"],
      "sector": "string",
      "evidence_quote": "verbatim from chapter"
    }
  ] or "Not Applicable",
  "related_theories": [
    {"theory": "name and source", "relevance": "explanation"}
  ] or "Not Applicable"
}
```

## KSP Reports (Pilot)

- `2009_VNM_Supporting the Establishment of Vietnam's 2011-20 Socio-economic Development Strategy_E.pdf`
- `2014_SLV_Developing an Innovation Ecosystem... Plastics Pharmaceutical and Cosmetics and Textile Industries.pdf`
- `2023_KAZ_Project on Extending the Life of Old Power Plants...Kazakhstan.pdf`
- `2023_QAT_Climate Smart Agriculture and Indoor Farming in Qatar.pdf`

## Project Structure

```
KM4D_v0/                              # Git repo (local)
  ksp_pilot_complete.ipynb             # Main Colab notebook
  Taxonomy_20250925.pdf                # Taxonomy reference
  CLAUDE.md                            # This file
  README.md

Google Drive (KM4D_v0/):              # Runtime data
  data/
    raw/
      ksp_reports/                     # 4 KSP PDF reports
      textbooks/                       # 2 textbook PDFs
    processed/
      chapter_summaries.json           # Chapter outlines from extraction
      textbook_chunks.json             # Processed textbook chunks
    results/
      chapter_analysis.json            # Main output
      sector_distribution.png
      knowledge_type_distribution.png
      sector_knowledge_heatmap.png
      network_*.png                    # Per-report theory-practice networks
  vector_db/                           # ChromaDB persistence (textbooks only)
```

## Setup Instructions

### Google Colab (Recommended)

1. Open `ksp_pilot_complete.ipynb` in Colab
2. Add `ANTHROPIC_API_KEY` to Colab Secrets (key icon in sidebar)
3. Upload KSP report PDFs to `MyDrive/KM4D_v0/data/raw/ksp_reports/`
4. Upload textbook PDFs to `MyDrive/KM4D_v0/data/raw/textbooks/`
5. Run all cells sequentially

### API Key

```python
from google.colab import userdata
ANTHROPIC_API_KEY = userdata.get('ANTHROPIC_API_KEY')
```

## Troubleshooting

### Chapter Extraction Issues
- **Too few chapters:** Font-size heuristics may not work well for some PDFs. Check `_identify_header_thresholds()` output.
- **Too many chapters:** Lower threshold may capture non-header text. Increase minimum text length filter.

### API Errors
- **Rate limits:** The pipeline has 1-second pauses between calls. Increase `time.sleep()` if needed.
- **JSON parse errors:** The response is cleaned of markdown fences. Check `raw_response` in error entries.
- **Context too long:** Chapters are truncated at 15,000 chars. Adjust `max_content_chars` in `_build_prompt()`.

### ChromaDB Issues
- **Textbooks not indexed:** Check `textbook_store.get_stats()`. If 0 chunks, re-run indexing cell.
- **Persistence:** ChromaDB persists to Google Drive. Data survives Colab restarts.

## Scaling Plan

After successful pilot:
1. Process all 566 KSP reports in batches
2. Add remaining 2 textbooks
3. Batch processing with checkpoint/resume support
4. Comprehensive cross-report analysis

## References

- Mokyr (2002) on propositional vs. prescriptive knowledge
- Ohno (2023) on translative adaptation
- Lewis et al. (2020) "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Perkins (2012) *Economics of Development*
- Todaro (2012) *Economic Development*

---

**Last Updated:** 2026-02-08
**Status:** Pilot implementation ready
**Implementation:** `ksp_pilot_complete.ipynb` (Google Colab)
