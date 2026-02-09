# KSP Development Knowledge Extraction

**Chapter-Level Classification & Policy Extraction using LLM + RAG**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research methodology for extracting and classifying development knowledge from Korea's Knowledge Sharing Program (KSP) reports at the chapter level, linking Korean policy experiences with theoretical frameworks from development economics textbooks.

---

## Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Output Schema](#output-schema)
- [Installation](#installation)

---

## Overview

This project analyzes **KSP advisory reports** at the chapter/sub-chapter level, performing four tasks per chapter:

1. **Taxonomy Classification** -- tags sectors & keywords from a 6-sector, ~140-keyword development cooperation taxonomy
2. **Knowledge Type Classification** -- classifies content as one of 4 knowledge types
3. **Korean Policy Extraction** -- extracts structured policy information (name, year, organization, instruments, etc.)
4. **Theory Linking** -- matches related theories from development economics textbooks via RAG

**Current Phase:** Pilot study with 4 KSP reports + 2 textbooks

**Platform:** Google Colab

### How It Works

```
KSP Report PDFs                   Textbook PDFs
      |                                 |
      v                                 v
  ChapterExtractor               Chunk & Embed
  (font-size heuristics)         (sentence-transformers)
      |                                 |
      v                                 v
  Chapters/Sub-chapters          ChromaDB Vector Store
      |                                 |
      +----------------+----------------+
                        |
                        v
            Combined LLM Prompt (Claude)
            per chapter:
              1. Taxonomy classification
              2. Knowledge type
              3. Korean policy extraction
              4. Theory linking (RAG context)
                        |
                        v
              chapter_analysis.json
                        |
                        v
              Visualizations & Analysis
```

**Key design decisions:**
- Chapters are read directly into the LLM context (no KSP chunking/RAG needed)
- Textbook RAG is kept for theory linking
- Single combined LLM call per chapter minimizes API costs
- Full taxonomy embedded in prompt (~3K tokens)

---

## Key Features

### Chapter-Level Analysis
- **ChapterExtractor** uses font-size heuristics to detect chapter/sub-chapter boundaries
- No chunking or embedding of KSP reports -- full chapter text sent to LLM
- Preserves document structure and context

### Development Cooperation Taxonomy
- 6 sectors, 4-level hierarchy, ~140 keywords from the official taxonomy
- Chapters mapped to specific sectors and keywords
- Multi-sector tagging supported

### 4 Knowledge Types
- Contextual background and situation analysis
- Policy implementation and coordinating mechanism
- Technical methodology and analytical framework
- Recommendations and future directions

### Korean Policy Extraction
- Structured fields: policy_name, year_initiated, organization, challenge_addressed, policy_instruments, sector
- Evidence quotes required (verbatim from source)
- "Not Applicable" when no Korean policies found in a chapter

### Theory Linking via Textbook RAG
- Textbooks indexed in ChromaDB (~7K chunks)
- Per-chapter query retrieves relevant theory passages
- LLM links policies to theoretical frameworks

### Visualizations
- Sector distribution charts
- Knowledge type distribution
- Sector x Knowledge type heatmap
- Theory-practice network graphs per report

---

## Quick Start

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yilmajung/KM4D_v0/blob/main/ksp_pilot_complete.ipynb)

### 2. Setup

```python
# Mount Google Drive (follow prompts)
from google.colab import drive
drive.mount('/content/drive')

# Add API key to Colab Secrets (key icon in sidebar)
# Name: ANTHROPIC_API_KEY
# Value: sk-ant-...
```

### 3. Upload PDFs

Upload to Google Drive:
- 4 KSP reports -> `MyDrive/KM4D_v0/data/raw/ksp_reports/`
- 2 textbooks -> `MyDrive/KM4D_v0/data/raw/textbooks/`

### 4. Run Pipeline

Run all cells sequentially:
- **Section 1-3:** Setup, config, taxonomy
- **Section 4:** Chapter extraction from KSP reports
- **Section 5:** Textbook vector store (index once)
- **Section 6:** Extract chapters from all reports
- **Section 7-8:** LLM classification + extraction
- **Section 9:** Visualization
- **Section 10:** Summary

### 5. View Results

```python
import json
with open('/content/drive/MyDrive/KM4D_v0/data/results/chapter_analysis.json') as f:
    results = json.load(f)

print(f"Analyzed {len(results)} chapters")
print(json.dumps(results[0], indent=2))
```

---

## Architecture

### Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Platform** | Google Colab | Free compute |
| **Storage** | Google Drive | Persistent data across sessions |
| **PDF Processing** | PyMuPDF (fitz) | Chapter extraction via font-size heuristics |
| **Textbook Chunking** | LangChain RecursiveCharacterTextSplitter | 768-token chunks |
| **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 | 384-dim vectors |
| **Vector DB** | ChromaDB | Textbook collection only |
| **LLM** | Anthropic Claude Sonnet 4 | Combined classification + extraction |
| **Visualization** | Matplotlib, Seaborn, NetworkX | Charts and network graphs |

### Data Flow

```
Phase 1: Chapter Extraction
  KSP PDFs -> ChapterExtractor -> chapters/sub-chapters (font-size heuristics)

Phase 2: Textbook Indexing (run once)
  Textbook PDFs -> LangChain chunking -> sentence-transformers -> ChromaDB

Phase 3: Combined Analysis (per chapter)
  For each chapter:
    1. Query textbook ChromaDB for theory context
    2. Build combined prompt (taxonomy + knowledge types + chapter text + theory context)
    3. Call Claude API -> parse JSON response
    4. Save to chapter_analysis.json

Phase 4: Visualization
  chapter_analysis.json -> sector charts, heatmaps, network graphs
```

---

## Output Schema

Each chapter produces a JSON entry like:

```json
{
  "report_id": "2023_QAT_Climate Smart Agriculture...",
  "chapter_title": "3. Smart Agricultural Policy in Korea",
  "chapter_level": 2,
  "page_start": 45,
  "page_end": 62,
  "content_length": 8200,
  "taxonomy_classification": {
    "sectors": [{
      "sector": "(4) Production & Trade",
      "sub_sector_l1": "Agriculture, Forestry & Fisheries",
      "sub_sector_l2": "Agricultural Development",
      "keywords": ["Agricultural Policy & Administration", "Sustainable Agriculture"]
    }],
    "knowledge_type": "Policy implementation and coordinating mechanism",
    "confidence": "high",
    "reasoning": "Chapter describes Korea's smart agriculture policy framework..."
  },
  "korean_policies": [
    {
      "policy_name": "Smart Farm Innovation Valley Project",
      "year_initiated": 2018,
      "organization": "Ministry of Agriculture, Food and Rural Affairs",
      "challenge_addressed": "Aging farming population and low agricultural productivity",
      "policy_instruments": ["Technology demonstration farms", "Training programs", "R&D subsidies"],
      "sector": "Agriculture",
      "evidence_quote": "In 2018, the Korean government launched the Smart Farm Innovation Valley..."
    }
  ],
  "related_theories": [
    {
      "theory": "Agricultural transformation and structural change (Todaro Ch. 9)",
      "relevance": "Korea's shift from traditional to technology-intensive agriculture..."
    }
  ]
}
```

When no Korean policies are found: `"korean_policies": "Not Applicable"`
When no theory links are found: `"related_theories": "Not Applicable"`

---

## Installation

### Option A: Google Colab (Recommended)

No installation needed. Open the notebook and run all cells.

**Requirements:**
- Google account
- Anthropic API key ([get one here](https://console.anthropic.com/))

### Option B: Local Setup

```bash
git clone https://github.com/yilmajung/KM4D_v0.git
cd KM4D_v0

python -m venv venv
source venv/bin/activate

pip install pymupdf pdfplumber sentence-transformers chromadb anthropic \
  pandas numpy scikit-learn matplotlib seaborn plotly networkx

export ANTHROPIC_API_KEY="sk-ant-..."
```

---

## KSP Reports Used (Pilot)

| Report | Country | Topic |
|--------|---------|-------|
| 2009_VNM | Vietnam | Socio-economic Development Strategy 2011-20 |
| 2014_SLV | El Salvador | Innovation Ecosystem (Plastics, Pharma, Textiles) |
| 2023_KAZ | Kazakhstan | Extending Life of Old Power Plants |
| 2023_QAT | Qatar | Climate Smart Agriculture & Indoor Farming |

## Textbooks Used (Pilot)

- Perkins (2012) - *Economics of Development*
- Todaro (2012) - *Economic Development*

---

## Project Structure

```
KM4D_v0/
  ksp_pilot_complete.ipynb     # Main Colab notebook
  Taxonomy_20250925.pdf        # Taxonomy reference
  CLAUDE.md                    # Project instructions
  README.md                    # This file

Google Drive (KM4D_v0/):
  data/
    raw/
      ksp_reports/             # 4 KSP PDF reports
      textbooks/               # 2 textbook PDFs
    processed/
      chapter_summaries.json   # Extracted chapter outlines
      textbook_chunks.json     # Processed textbook chunks
    results/
      chapter_analysis.json    # Main output
      sector_distribution.png
      knowledge_type_distribution.png
      sector_knowledge_heatmap.png
      network_*.png            # Per-report theory-practice networks
  vector_db/                   # ChromaDB persistence (textbooks)
```

---

**Last Updated:** 2026-02-08
