# KSP Development Knowledge Extraction

**Linking Theory and Practice in Development Economics using LLM + RAG**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A research methodology for systematically extracting and linking development policies from Korea's Knowledge Sharing Program (KSP) reports with theoretical frameworks from development economics textbooks.

**Key Idea:** Systematic theory-practice mapping in development economics at scale using dual-collection RAG architecture.

---

## Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Research Questions](#research-questions)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Expected Outputs](#expected-outputs)
- [Documentation](#documentation)
- [Citation](#citation)
- [License](#license)

---

## Overview

This project extracts **prescriptive knowledge** (policies, programs, implementation details) from 566 KSP advisory reports and links them to **propositional knowledge** (theories, frameworks) from development economics textbooks.

**Current Phase:** Pilot study with 4 KSP reports + 2 textbooks

**Methodology:** Open-source LLM + Retrieval-Augmented Generation (RAG) with dual vector databases

**Platform:** Google Colab (A100 GPU)

### What This Project Does

```
KSP Reports (Practice)          Dev Economics Textbooks (Theory)
        ↓                                      ↓
   Extract policies                   Extract concepts
        ↓                                      ↓
   ChromaDB Collection                ChromaDB Collection
        ↓                                      ↓
        └──────────────┬──────────────────────┘
                       ↓
              Cross-Query & Link
                       ↓
         ┌─────────────────────────┐
         │  Policy Database with   │
         │  Theory Links           │
         └─────────────────────────┘
                       ↓
         Analysis & Visualization
```

---

## Key Features

### Dual-Collection RAG Architecture
- **KSP Collection:** 566 policy reports (20 years, 1,513 topics)
- **Textbook Collection:** 4 development economics textbooks
- **Cross-querying:** Automatic theory-practice linking

### Extraction
- Policy name, year, organization, instruments
- Evidence quotes (grounded in source documents)
- Related theoretical concepts
- Development stage classification

### Evaluation
- Precision, Recall, F1 metrics
- Gold standard comparison
- Error analysis and categorization

### Visualizations
- Sector distribution charts
- Theory-practice network graphs
- Temporal trend analysis
- Coverage gap identification

### Persistence & Scalability
- ChromaDB persists to Google Drive
- Process once, query many times
- Incremental processing support
- Scales from 4 to 566 reports

---

## Quick Start

### 1. Open in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

```python
# Create new notebook, paste the complete code from ksp_pilot_complete.py
```

### 2. Setup

```python
# Mount Google Drive (follow prompts)
from google.colab import drive
drive.mount('/content/drive')

# Add API key to Colab Secrets
# Name: ANTHROPIC_API_KEY
# Value: sk-ant-...
```

### 3. Upload PDFs

Navigate in Colab file browser:
- Upload 4 KSP reports → `MyDrive/KSP_Pilot/data/raw/ksp_reports/`
- Upload 2 textbooks → `MyDrive/KSP_Pilot/data/raw/textbooks/`

**Naming conventions:**
- KSP: `2015_VNM_Industrial_Policy.pdf`
- Textbooks: `Perkins_2013.pdf`

### 4. Run Pipeline (30-45 minutes)

```python
# Execute the complete pipeline
# Section 7: Index documents (10 min)
# Section 10: Extract policies (15 min)  
# Section 12: Analyze results (5 min)
```

### 5. View Results

```python
# Check extracted policies
import json
with open('/content/drive/MyDrive/KSP_Pilot/data/results/extracted_policies.json') as f:
    policies = json.load(f)
    
print(f"Extracted {len(policies)} policies")
print(json.dumps(policies[0], indent=2))
```

**Expected output:** 30-50 policies with theory links, sector classification, and evidence quotes

---

## Research Questions

This methodology addresses three core questions:

### RQ1: How to systematically link theory with practice?
**Answer:** Dual-collection RAG with cross-querying
- Extract policies from KSP reports
- Extract concepts from textbooks
- Use semantic similarity to connect them

### RQ2: What gaps exist between theory and practice?
**Answer:** Theory coverage analysis
- Which theories are well-implemented? (many policies)
- Which theories are underutilized? (few policies)
- Which practices lack theoretical foundation?

### RQ3: Do policies cluster by sectors or other dimensions?
**Answer:** Multi-dimensional analysis
- Traditional: Sectoral categories (manufacturing, finance, etc.)
- Novel: Governance mechanisms, development stages, policy instruments
- Emergent: Data-driven clustering reveals new patterns

---

## Architecture

### Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Platform** | Google Colab | Free compute with GPU acceleration (or A100) |
| **Storage** | Google Drive | Persistent data across sessions |
| **PDF Processing** | PyMuPDF, pdfplumber | Text extraction with structure |
| **Chunking** | LangChain | Semantic segmentation (512/768 tokens) |
| **Embeddings** | sentence-transformers | 384-dim vectors (all-MiniLM-L6-v2) |
| **Vector DB** | ChromaDB | Dual collections with metadata |
| **LLM** | Anthropic Claude Sonnet 4 | Structured extraction + linking |
| **Evaluation** | scikit-learn | Precision, Recall, F1 |
| **Visualization** | Matplotlib, NetworkX | Charts and network graphs |

### ChromaDB Collections

**Collection 1: KSP Reports**
```python
{
  'collection': 'ksp_reports_pilot',
  'chunks': ~320,  # 4 reports × 80 chunks
  'metadata': ['year', 'country', 'sector', 'filename']
}
```

**Collection 2: Textbooks**
```python
{
  'collection': 'textbooks_pilot',
  'chunks': ~400,  # 2 books × 200 chunks
  'metadata': ['textbook', 'chapter', 'concept']
}
```

### Data Flow

```
PDFs → Extract Text → Create Chunks → Generate Embeddings
  ↓
Store in ChromaDB (persist to Google Drive)
  ↓
Query both collections → Retrieve relevant passages
  ↓
Send to Claude API → Extract structured policies
  ↓
Link policies to theories → Save as JSON
  ↓
Analyze & Visualize → Generate insights
```

---

## Installation

### Option A: Google Colab

No installation needed! Everything runs in browser.

**Requirements:**
- Google account
- Anthropic API key ([get one here](https://console.anthropic.com/))

### Option B: Local Setup

```bash
# Clone repository
git clone https://github.com/yilmajung/ksp-knowledge-extraction.git
cd ksp-knowledge-extraction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install pymupdf pdfplumber sentence-transformers chromadb anthropic \
  pandas numpy scikit-learn matplotlib seaborn plotly networkx umap-learn

# Set API key
export ANTHROPIC_API_KEY="sk-ant-..."
```
<!-- 
---

## Usage

### Basic Workflow

```python
# The complete implementation is in ksp_pilot_complete.py
# Just run it in Google Colab!

# Phase 1: Index documents (run once)
ksp_chunks = process_and_index_ksp_reports()
textbook_chunks = process_and_index_textbooks()

# Phase 2: Extract policies (run many times)
extracted_policies = extract_all_policies()

# Phase 3: Analyze
policy_df = analyze_policies(extracted_policies)
```

### Query ChromaDB Directly

```python
# Search KSP reports
results = ksp_store.search(
    query="export processing zones industrial policy",
    n_results=5,
    filter_dict={"sector": "Manufacturing"}
)

# Search textbooks
theory_results = textbook_store.search(
    query="special economic zones development theory",
    n_results=3
)
```

### Custom Analysis

```python
# Filter by sector
manufacturing_policies = [
    p for p in extracted_policies 
    if p['sector'] == 'Manufacturing'
]

# Theory coverage analysis
from collections import Counter
theory_counts = Counter(
    p['related_theory'] 
    for p in extracted_policies 
    if p.get('related_theory')
)

print("Top 10 theories in practice:")
for theory, count in theory_counts.most_common(10):
    print(f"  {theory}: {count} policies")
``` -->