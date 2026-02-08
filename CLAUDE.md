# KSP Development Knowledge Extraction: Open-Source LLM + RAG

## Project Overview

Extract and link development knowledge from KSP (Knowledge Sharing Program) advisory reports and development economics textbooks using an open-source LLM + RAG (Retrieval-Augmented Generation) pipeline. This is a research methodology for analyzing 20 years of development cooperation knowledge and systematically connecting theory with practice.

**Research Questions:**
1. Can we systematically extract prescriptive knowledge (policies, programs) from KSP development experience reports?
2. How do these extracted policies relate to propositional knowledge (development economics theory from textbooks)?
3. What alternative categorizations of development knowledge emerge beyond traditional sectoral classifications?

## Current Phase: Small-Scale Pilot

**Scope:**
- **4 KSP advisory reports** (your choice from 566 available)
- **2 development economics textbooks** (e.g., Perkins 2013, Todaro 2020)
- **Dual-collection RAG** with ChromaDB
- **Theory-practice linking** using cross-collection queries

**Goals:**
- Validate the technical approach (indexing, retrieval, extraction)
- Test theory-practice linking methodology
- Optimize extraction prompts
- Establish baseline accuracy (target F1 ≥ 0.70)
- Identify challenges before scaling to 566 reports + 4 textbooks

## Technical Stack

**Environment:**
- **Platform:** Google Colab (with T4 GPU for faster embeddings)
- **Storage:** Google Drive (for persistence across sessions)
- **Language:** Python 3.10+

**Core Components:**
- **PDF Processing:** PyMuPDF (fitz) + pdfplumber for text extraction
- **Text Chunking:** LangChain RecursiveCharacterTextSplitter
  - KSP reports: 512 tokens per chunk
  - Textbooks: 768 tokens per chunk
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
  - 384-dimensional vectors
  - Free, runs locally (or on Colab GPU)
- **Vector Database:** ChromaDB for embedding storage and retrieval
  - Dual collections: KSP reports + Textbooks
  - Persists to Google Drive
  - Supports metadata filtering
- **LLM:** Anthropic Claude API (Claude Sonnet 4)
  - For structured policy extraction
  - For theory-practice linking
  - Cost: ~$0.22 for pilot, ~$28 for full 566 reports
- **Evaluation:** Custom metrics (Precision, Recall, F1) + error analysis
- **Visualization:** NetworkX, Matplotlib, Seaborn for analysis

## Project Structure

```
Google Drive/MyDrive/KSP_Pilot/
├── data/
│   ├── raw/
│   │   ├── ksp_reports/           # Upload 4 KSP PDF reports here
│   │   │   ├── 2015_VNM_Industrial_Policy.pdf
│   │   │   ├── 2018_ETH_Financial_Inclusion.pdf
│   │   │   ├── 2020_IDN_Digital_Economy.pdf
│   │   │   └── 2022_KEN_Agriculture.pdf
│   │   └── textbooks/              # Upload 2 textbook PDFs here
│   │       ├── Perkins_2013.pdf
│   │       └── Todaro_2020.pdf
│   ├── processed/
│   │   ├── ksp_chunks.json        # Processed KSP chunks with metadata
│   │   └── textbook_chunks.json   # Processed textbook chunks
│   ├── gold_standard/             # Manual annotations for validation
│   │   └── annotations.json       # (Optional - create for evaluation)
│   └── results/
│       ├── extracted_policies.json         # Main output!
│       ├── evaluation_results.json         # If gold standard exists
│       ├── sector_distribution.png         # Visualizations
│       └── theory_practice_network.png
├── vector_db/                     # ChromaDB persistence (auto-created)
│   ├── chroma.sqlite3             # Main database
│   └── [collection_uuid]/         # Data files for each collection
└── notebooks/                     # (Optional - for additional analysis)
    └── ksp_pilot_complete.ipynb   # Main Colab notebook

Note: All paths relative to Google Drive. Data persists across Colab sessions.
```

## Setup Instructions

### Quick Start (Recommended)

**Option A: Use the Provided Complete Implementation**

1. **Open Google Colab:** https://colab.research.google.com/
2. **Create new notebook:** File → New notebook
3. **Copy the complete code:**
   - Open `ksp_pilot_complete.py` (provided in your files)
   - Copy entire content
   - Paste into a Colab code cell
4. **Enable GPU (optional but recommended):**
   - Runtime → Change runtime type → T4 GPU
5. **Run the cell** - follow prompts for:
   - Google Drive mounting (authorize access)
   - API key (add to Colab secrets 🔑 or enter when prompted)
6. **Upload PDFs** to created Google Drive folders
7. **Run again** - complete pipeline executes automatically

**Estimated time:** 30-45 minutes total
**Estimated cost:** ~$0.22 (Anthropic API)

---

### Manual Setup (If You Want to Build From Scratch)

#### 1. Environment Setup

**Google Colab:**
```python
# Install required packages
!pip install -q pymupdf pdfplumber sentence-transformers chromadb anthropic \
  pandas numpy scikit-learn matplotlib seaborn plotly networkx umap-learn

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create project directory
import os
project_dir = '/content/drive/MyDrive/KSP_Pilot'
os.makedirs(f'{project_dir}/data/raw/ksp_reports', exist_ok=True)
os.makedirs(f'{project_dir}/data/raw/textbooks', exist_ok=True)
os.makedirs(f'{project_dir}/data/processed', exist_ok=True)
os.makedirs(f'{project_dir}/data/results', exist_ok=True)
os.makedirs(f'{project_dir}/vector_db', exist_ok=True)
```

**Local Setup (Alternative):**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pymupdf pdfplumber sentence-transformers chromadb anthropic \
  pandas numpy scikit-learn matplotlib seaborn plotly networkx umap-learn jupyter
```

#### 2. API Key Configuration

**For Google Colab (Recommended):**
1. Click **🔑 Secrets** icon in left sidebar
2. Add new secret:
   - Name: `ANTHROPIC_API_KEY`
   - Value: Your API key (starts with `sk-ant-...`)
   - Toggle "Notebook access" ON

**Access in code:**
```python
from google.colab import userdata
ANTHROPIC_API_KEY = userdata.get('ANTHROPIC_API_KEY')
```

**For Local:**
```python
import os
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
# Or use python-dotenv with .env file
```

#### 3. Data Preparation

**Upload PDFs to Google Drive:**

**KSP Reports** (`data/raw/ksp_reports/`):
- Naming: `YYYY_CCC_Title.pdf`
- Examples:
  - `2015_VNM_Industrial_Policy.pdf`
  - `2018_ETH_Financial_Inclusion.pdf`
  - `2020_IDN_Digital_Economy.pdf`
  - `2022_KEN_Agriculture_Modernization.pdf`

**Textbooks** (`data/raw/textbooks/`):
- Naming: `Author_Year.pdf`
- Examples:
  - `Perkins_2013.pdf` (Economics of Development)
  - `Todaro_2020.pdf` (Economic Development)

**How to upload in Colab:**
1. Click 📁 Files icon in left sidebar
2. Navigate to the folder
3. Right-click → Upload
4. Select PDFs from your computer

## Configuration

### Using the Provided Implementation

The `ksp_pilot_complete.py` file includes a `Config` dataclass with all parameters pre-configured:

```python
@dataclass
class Config:
    # Directories (auto-configured for Google Drive)
    project_dir: str = "/content/drive/MyDrive/KSP_Pilot"
    
    # Chunking parameters
    ksp_chunk_size: int = 512          # Tokens per chunk (KSP reports)
    textbook_chunk_size: int = 768     # Tokens per chunk (textbooks - larger for theory)
    chunk_overlap: int = 50
    
    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM API
    llm_model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.1           # Low for factual extraction
    max_tokens: int = 4000
    
    # Retrieval parameters
    ksp_top_k: int = 5                 # Number of KSP chunks to retrieve
    textbook_top_k: int = 3            # Number of textbook chunks to retrieve
    
    # ChromaDB Collections
    ksp_collection: str = "ksp_reports_pilot"
    textbook_collection: str = "textbooks_pilot"
```

**To customize:** Edit values in the Config dataclass (Section 2 of the notebook).

**Key parameters to tune during pilot:**
- `ksp_chunk_size`: Try 256, 512, 768 to find optimal
- `ksp_top_k`: Increase to 10 if missing context
- `temperature`: Keep at 0.1 for factual extraction
- `textbook_top_k`: Increase if you need more theoretical context

### Manual Configuration (Advanced)

If building from scratch, create these YAML config files:
```yaml
# Schema for extracted policy information
policy_schema:
  policy_name:
    type: string
    required: true
    description: "Official title or clear description of policy/program"
  
  year_initiated:
    type: integer
    required: false
    description: "Year the policy was initiated"
  
  organization:
    type: string
    required: false
    description: "Government ministry, agency, or organization responsible"
  
  challenge_addressed:
    type: string
    required: false
    description: "Development problem or challenge the policy addressed"
  
  policy_instruments:
    type: array
    required: false
    description: "Specific mechanisms, tools, or interventions used"
  
  sector:
    type: string
    required: false
    description: "Economic sector (manufacturing, finance, agriculture, etc.)"
  
  development_stage:
    type: string
    required: false
    enum: ["early_industrialization", "middle_income", "advanced"]
    description: "Stage of development when policy was implemented"
  
  evidence_quote:
    type: string
    required: true
    description: "Direct quote from document supporting this extraction"
  
  source_page:
    type: integer
    required: false
    description: "Page number where policy is mentioned"
```

Create `configs/llm_config.yaml`:
```yaml
# LLM configuration
model:
  name: "llama3.1:8b"  # Change to llama3.1:70b for production
  temperature: 0.1      # Low temperature for factual extraction
  max_tokens: 2000      # Maximum tokens in response
  
inference:
  timeout: 120          # Timeout in seconds
  retry_attempts: 3     # Number of retries on failure
  
logging:
  level: "INFO"
  save_prompts: true    # Save all prompts for debugging
  save_responses: true  # Save all responses
```

Create `configs/rag_config.yaml`:
```yaml
# RAG configuration
chunking:
  strategy: "section_based"  # section_based | fixed_size | semantic
  chunk_size: 512           # tokens per chunk
  chunk_overlap: 50         # token overlap between chunks
  preserve_sections: true   # Keep section boundaries intact
  
embeddings:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384            # Embedding dimension
  normalize: true           # L2 normalize embeddings
  
vector_store:
  type: "chromadb"
  collection_name: "ksp_reports_pilot"
  persist_directory: "./data/vector_db"
  
retrieval:
  top_k: 5                  # Number of chunks to retrieve per query
  similarity_threshold: 0.7 # Minimum cosine similarity
  use_reranking: false      # Use cross-encoder reranking (slower but better)
  
queries:
  # Multiple queries to retrieve comprehensive context
  - "Korean government policy program initiative"
  - "ministry organization implementation agency"
  - "economic development challenge problem solution"
  - "policy instrument mechanism tool intervention"
```

## Implementation Overview

### Complete Implementation Provided

The `ksp_pilot_complete.py` file is a **ready-to-run** implementation with 13 sections:

**Sections 1-6: Core Components**
1. Setup & Installation
2. Configuration
3. PDF Processing (PyMuPDF)
4. Semantic Chunking (LangChain)
5. Embedding & Vector Database (sentence-transformers + ChromaDB)
6. Dual Vector Store Initialization (KSP + Textbooks)

**Sections 7-10: Main Pipeline**
7. **Document Indexing** - Process and add PDFs to ChromaDB (run once)
8. LLM Extraction Setup (Claude API)
9. Extraction Pipeline with Theory Linking
10. **Run Extraction** - Extract policies from all reports (query many times)

**Sections 11-13: Analysis**
11. Evaluation (if gold standard exists)
12. Analysis & Visualization (sector distribution, theory-practice networks)
13. Summary & Results

### How to Use

**Option 1: Run Everything (Recommended for First Time)**
```python
# Copy entire ksp_pilot_complete.py into Colab
# Run the cell
# Follow prompts and upload PDFs
# Complete pipeline executes automatically
```

**Option 2: Run Section by Section (For Understanding/Debugging)**
```python
# Copy sections one at a time
# Section 7: Index documents (once)
# Section 10: Extract policies (many times)
# Section 12: Analyze results
```

**Option 3: Modify and Extend**
```python
# Use the provided code as a template
# Customize specific functions
# Add your own analysis
```

---

## Key Implementation Details

### Dual-Collection Architecture

**Why two collections?**
- **KSP Collection:** Prescriptive knowledge (what was done in practice)
- **Textbook Collection:** Propositional knowledge (theory and frameworks)
- **Benefit:** Cross-query to link theory ↔ practice

**Implementation:**
```python
# Initialize both collections
ksp_store = VectorStore(collection_name="ksp_reports_pilot")
textbook_store = VectorStore(collection_name="textbooks_pilot")

# Both persist to same ChromaDB directory
persist_directory = "/content/drive/MyDrive/KSP_Pilot/vector_db"
```

### Chunking Strategy

**Different sizes for different sources:**
```python
# KSP reports: 512 tokens
# Why: Policy descriptions typically 1-3 paragraphs
chunker_ksp = SemanticChunker(chunk_size=512, chunk_overlap=50)

# Textbooks: 768 tokens
# Why: Theoretical explanations need more context
chunker_textbook = SemanticChunker(chunk_size=768, chunk_overlap=100)
```

### Retrieval Strategy

**Multi-query approach for comprehensive context:**
```python
# Query KSP collection with multiple targeted queries
ksp_queries = [
    "Korean government policy program initiative",
    "ministry organization agency institution",
    "economic development challenge problem solution",
    "policy instrument mechanism tool intervention"
]

# Aggregate all results
all_context = set()
for query in ksp_queries:
    results = ksp_store.search(query, n_results=5)
    all_context.update(results['documents'][0])
```

**Cross-collection theory linking:**
```python
# After retrieving policy context, query textbooks
theory_results = textbook_store.search(
    query="development policy economic growth industrial policy",
    n_results=3
)

# Send both contexts to LLM for extraction
policies = extractor.extract_from_contexts(
    ksp_context=ksp_context,
    theory_context=theory_context
)
```

### Extraction Prompt

**Structured output with theory linking:**
```python
extraction_prompt = """
CONTEXT FROM KSP REPORT:
{ksp_context}

RELATED THEORETICAL CONCEPTS (from textbooks):
{theory_context}

Extract ALL policies mentioned, providing:
1. policy_name
2. year_initiated
3. organization
4. challenge_addressed
5. policy_instruments
6. sector
7. development_stage
8. evidence_quote (REQUIRED - verbatim from KSP)
9. related_theory (from textbook context)

Return JSON array...
"""
```

---

## Workflow: Step-by-Step

### Phase 1: Document Indexing (Run Once)

**What happens:**
1. PDFs → Text extraction → Sections
2. Sections → Semantic chunks (512/768 tokens)
3. Chunks → Embeddings (384-dim vectors)
4. Embeddings + Metadata → ChromaDB collections

**Code (Section 7):**
```python
# Process KSP reports
ksp_chunks = process_and_index_ksp_reports()
# Output: ~320 chunks (4 reports × 80 chunks avg)

# Process textbooks
textbook_chunks = process_and_index_textbooks()
# Output: ~400 chunks (2 books × 200 chunks avg)

# Check status
print(ksp_store.get_stats())
print(textbook_store.get_stats())
```

**Time:** 5-10 minutes (with GPU)

### Phase 2: Policy Extraction (Run Many Times)

**What happens:**
1. For each report:
   - Query KSP collection (5 queries × 5 results = 25 chunks)
   - Query textbook collection (1 query × 3 results = 3 chunks)
   - Combine contexts
   - Send to Claude API
   - Parse JSON response
   - Link policies to theories
2. Save all policies to JSON

**Code (Section 10):**
```python
extracted_policies = extract_all_policies()
# Output: extracted_policies.json with all policies
```

**Time:** 5-10 minutes (API dependent)

### Phase 3: Analysis (Run Once)

**What happens:**
1. Load extracted policies
2. Calculate statistics (sector distribution, theory coverage)
3. Generate visualizations
4. Evaluate against gold standard (if exists)

**Code (Section 12):**
```python
policy_df = analyze_policies(extracted_policies)
# Output: PNG charts, network graphs, summary stats
```

**Time:** 2-3 minutes

---

## Understanding ChromaDB in This Project

### Initialization

```python
client = chromadb.Client(Settings(
    persist_directory="/content/drive/MyDrive/KSP_Pilot/vector_db",
    anonymized_telemetry=False
))
```

**Key points:**
- ✅ Persists to Google Drive (survives Colab sessions)
- ✅ No need to call save() - automatic
- ✅ Next session: data loads automatically

### Adding Documents (Indexing Phase)

```python
collection.add(
    documents=["text chunk 1", "text chunk 2", ...],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
    metadatas=[
        {"year": 2015, "country": "VNM", "sector": "Manufacturing"},
        {"year": 2018, "country": "ETH", "sector": "Finance"}
    ],
    ids=["2015_VNM_s3_c2", "2018_ETH_s1_c5", ...]
)
```

**When:** Once during indexing (Section 7)

### Querying (Extraction Phase)

```python
results = collection.query(
    query_texts=["Korean industrial policy export"],
    n_results=5,
    where={"filename": "2015_VNM_Industrial_Policy"}  # Filter by metadata
)

# Access results
documents = results['documents'][0]     # Retrieved text chunks
metadatas = results['metadatas'][0]     # Metadata for each chunk
distances = results['distances'][0]      # Similarity scores (lower = better)
```

**When:** Many times during extraction (Section 10)

### Metadata Filtering

```python
# Simple filter
where={"sector": "Manufacturing"}

# Comparison
where={"year": {"$gte": 2015}}  # Year >= 2015

# Complex filter
where={
    "$and": [
        {"year": {"$gte": 2015}},
        {"sector": {"$in": ["Manufacturing", "Finance"]}}
    ]
}
```

**Use cases:**
- Search only one report: `where={"filename": "2015_VNM_..."}`
- Search by time period: `where={"year": {"$gte": 2015}}`
- Search by sector: `where={"sector": "Manufacturing"}`

---

## Detailed Task Breakdown (If Building Manually)

If you're not using the complete implementation and want to build components yourself:

### Task 1: PDF Processing (30 min)
**File:** Implemented in `PDFProcessor` class (Section 3)

**Key features:**
- Extracts text with structure preservation
- Identifies section headers by font size
- Handles both KSP reports and textbooks
- Extracts metadata from filename

**Test:** Process one PDF and verify sections extracted

### Task 2: Semantic Chunking (20 min)

**File:** Implemented in `SemanticChunker` class (Section 4)

**Key features:**
- Different chunk sizes for KSP (512) vs textbooks (768)
- Preserves section boundaries
- Attaches comprehensive metadata

**Test:** Verify chunks are reasonable size and preserve context

### Task 3: Vector Database Setup (20 min)

**File:** Implemented in `VectorStore` class (Section 5-6)

**Key features:**
- Dual collections (KSP + Textbooks)
- Automatic embedding generation
- Persists to Google Drive
- Supports metadata filtering

**Test:** Add sample chunks, verify retrieval works

### Task 4: LLM Extraction (30 min)

**File:** Implemented in `PolicyExtractor` class (Section 8)
**Key features:**
- Theory-aware extraction prompt
- Structured JSON output with validation
- Links policies to theoretical concepts
- Requires evidence quotes

**Test:** Extract from one report, verify quality

---

## Gold Standard Creation (Optional but Recommended)

### Why Create Gold Standard?

**Purpose:** Measure extraction accuracy objectively (Precision, Recall, F1)

**Effort:** 2-4 hours for 2 reports (1-2 people)

**Value:** Essential for validating methodology before scaling

### Process

1. **Select 2 reports** from your 4 pilot reports
2. **Manually extract** all policies following the same schema:

```json
{
  "source_report": "2015_VNM_Industrial_Policy",
  "annotator": "Your Name",
  "policies": [
    {
      "policy_name": "Export Processing Zones",
      "year_initiated": 1970,
      "organization": "Ministry of Trade and Industry",
      "challenge_addressed": "Attract FDI and boost exports",
      "policy_instruments": ["Tax incentives", "Duty-free imports"],
      "sector": "Manufacturing",
      "development_stage": "middle_income",
      "evidence_quote": "In 1970, Korea established...",
      "source_page": 45
    }
  ]
}
```

3. **Save as:** `data/gold_standard/annotations.json`
4. **Run evaluation** (Section 11 in notebook) - automatically calculates P/R/F1

### Annotation Guidelines

**Include:**
- ✅ Explicit policy mentions with clear names
- ✅ Programs with specific implementation details
- ✅ Initiatives with defined objectives

**Exclude:**
- ❌ Vague references ("the government helped")
- ❌ General recommendations without specifics
- ❌ Historical context not about specific policies

**Evidence quotes:** Must be verbatim from document

---

## Evaluation Metrics

### Automatic Evaluation (If Gold Standard Exists)

**Entity-Level Metrics:**
```python
# Did we find the right policies?
Precision = Correct extractions / Total extractions
Recall = Correct extractions / Total policies in gold standard  
F1 Score = 2 × (Precision × Recall) / (Precision + Recall)
```

**Attribute-Level Metrics:**
```python
# Are the fields accurate?
Year Accuracy = Correct years / Total policies
Organization Accuracy = Correct organizations / Total policies
Sector Accuracy = Correct sectors / Total policies
```

**Implementation:** Section 11 in `ksp_pilot_complete.py`

**Interpretation:**
- **F1 ≥ 0.80:** Excellent - ready to scale immediately
- **F1 = 0.70-0.79:** Good - minor refinements needed
- **F1 = 0.60-0.69:** Acceptable - significant refinement needed
- **F1 < 0.60:** Poor - major methodology revision required

### Manual Quality Checks (Always Do This)

Even without gold standard, review samples:

```python
# Check first 10 policies
for policy in extracted_policies[:10]:
    print(f"\nPolicy: {policy['policy_name']}")
    print(f"Evidence: {policy['evidence_quote'][:100]}...")
    
    # Manual checks:
    # - Does evidence quote exist in source PDF?
    # - Is policy name accurate?
    # - Are policy instruments specific?
    # - Is related theory appropriate?
```

**Red flags:**
- Missing evidence quotes
- Generic policy names ("Government program")
- Empty policy_instruments lists
- Unrelated theory links

---

## Expected Outputs

### Main Output: Extracted Policies

**File:** `data/results/extracted_policies.json`

**Structure:**
```json
[
  {
    "policy_name": "Export Processing Zones",
    "year_initiated": 1970,
    "organization": "Ministry of Trade and Industry",
    "challenge_addressed": "Attract FDI, boost exports, create jobs",
    "policy_instruments": [
      "Tax holidays for 5 years",
      "Duty-free imports of raw materials",
      "Streamlined customs procedures",
      "Infrastructure provision"
    ],
    "sector": "Manufacturing",
    "development_stage": "middle_income",
    "evidence_quote": "In 1970, Korea established Export Processing Zones (EPZs) in Masan to attract foreign direct investment...",
    "related_theory": "Special Economic Zones theory - creating enclaves with preferential treatment to attract FDI (Perkins 2013, Chapter 18)",
    "source_report": "2015_VNM_Industrial_Policy"
  }
]
```

**Expected count:** 30-50 policies from 4 reports (8-12 per report average)

### Visualizations

**1. Sector Distribution** (`sector_distribution.png`)
- Bar chart showing policies by sector
- Identifies which sectors are most represented

**2. Theory-Practice Network** (`theory_practice_network.png`)
- Network graph connecting policies to theories
- Blue nodes = Policies
- Red nodes = Theories  
- Shows which theories are well-linked to practice

**3. Evaluation Report** (`evaluation_results.json` - if gold standard exists)
- Overall P/R/F1 scores
- Per-report breakdown
- Attribute-level accuracy

### Processed Data

**Chunks for Future Use:**
- `data/processed/ksp_chunks.json` - All KSP chunks with metadata
- `data/processed/textbook_chunks.json` - All textbook chunks

**ChromaDB Vector Database:**
- `vector_db/` - Persists across sessions
- Ready for additional queries without re-indexing

---

## Analysis Examples

### Query 1: Find All Policies in Manufacturing Sector

```python
# Filter extracted policies
manufacturing_policies = [
    p for p in extracted_policies 
    if p['sector'] == 'Manufacturing'
]

print(f"Found {len(manufacturing_policies)} manufacturing policies")

# What theories are linked?
theories = set(p.get('related_theory', 'None') for p in manufacturing_policies)
print(f"Related theories: {theories}")
```

### Query 2: Theory Coverage Analysis

```python
# Which theories are well-represented?
from collections import Counter

theory_counts = Counter(
    p.get('related_theory') 
    for p in extracted_policies 
    if p.get('related_theory')
)

print("Most common theories in practice:")
for theory, count in theory_counts.most_common(10):
    print(f"  {theory}: {count} policies")
```

### Query 3: Temporal Analysis

```python
# Policy evolution over time
import pandas as pd

df = pd.DataFrame(extracted_policies)
df = df[df['year_initiated'].notna()]

# Group by decade
df['decade'] = (df['year_initiated'] // 10) * 10
decade_counts = df.groupby('decade').size()

print("Policies by decade:")
print(decade_counts)
```

### Query 4: Find Policies Without Theory Links

```python
# Gap analysis: policies without theoretical foundation
unlinked = [
    p for p in extracted_policies 
    if not p.get('related_theory')
]

print(f"{len(unlinked)} policies without theory links ({len(unlinked)/len(extracted_policies)*100:.1f}%)")

# Are these in specific sectors?
sectors = Counter(p['sector'] for p in unlinked)
print(f"Unlinked policies by sector: {sectors}")
```

---

## Troubleshooting

### Issue 1: Low Extraction Quality (F1 < 0.60)

**Symptoms:**
- Many policies missed (low recall)
- Many hallucinated policies (low precision)
- Generic policy names

**Solutions:**
1. **Adjust chunking:**
   ```python
   config.ksp_chunk_size = 768  # Increase for more context
   # Re-run Section 7 (indexing)
   ```

2. **Increase retrieval:**
   ```python
   config.ksp_top_k = 10  # Retrieve more chunks
   # Re-run Section 10 (extraction)
   ```

3. **Refine prompt:**
   - Add examples to extraction prompt
   - Emphasize evidence quotes
   - Be more specific about policy instruments

### Issue 2: ChromaDB Not Persisting

**Symptoms:**
- Collections empty after restarting Colab
- "Collection not found" errors

**Solutions:**
```python
# Verify persist_directory
client = chromadb.Client(Settings(
    persist_directory="/content/drive/MyDrive/KSP_Pilot/vector_db"  # Must be Drive
))

# Check if files exist
import os
print(os.listdir("/content/drive/MyDrive/KSP_Pilot/vector_db"))
# Should see: chroma.sqlite3 and collection folders
```

### Issue 3: API Rate Limits

**Symptoms:**
- "Rate limit exceeded" errors
- Slow extraction speed

**Solutions:**
```python
import time

# Add delay between reports
for report in reports:
    policies = extract_policies_from_report(report)
    time.sleep(5)  # 5-second pause
```

### Issue 4: Poor Theory Linking

**Symptoms:**
- Most policies have `related_theory: null`
- Theories don't match policies

**Solutions:**
1. **Verify textbooks indexed:**
   ```python
   print(textbook_store.get_stats())
   # Should show ~400 chunks
   ```

2. **Adjust theory query:**
   ```python
   # Make query more specific
   theory_query = f"{policy['sector']} development policy economic theory"
   ```

3. **Increase textbook retrieval:**
   ```python
   config.textbook_top_k = 5  # More theory context
   ```

### Issue 5: Out of Memory

**Symptoms:**
- "CUDA out of memory" during embedding

**Solutions:**
```python
# Reduce batch size
vector_store.add_documents(chunks, batch_size=16)  # Default is 32

# Or disable GPU
# Runtime → Change runtime type → None (CPU only)
# Slower but more stable
```

---

## Success Criteria for Pilot

### Minimum Viable Success

- ✅ Successfully index 4 KSP reports + 2 textbooks
- ✅ Extract 30+ policies total (avg 8-12 per report)
- ✅ 50%+ policies have evidence quotes
- ✅ 30%+ policies linked to theories
- ✅ No major errors in extraction process

### Good Success

- ✅ All of the above, plus:
- ✅ F1 score ≥ 0.70 (if gold standard created)
- ✅ Clear sector patterns visible
- ✅ Theory-practice network shows meaningful connections
- ✅ Lessons learned documented for scaling

### Excellent Success

- ✅ All of the above, plus:
- ✅ F1 score ≥ 0.80
- ✅ 60%+ policies linked to theories
- ✅ Novel insights about theory-practice gaps identified
- ✅ Ready to scale immediately to 566 reports

---

## Scaling Plan (After Successful Pilot)

### If F1 ≥ 0.70: Ready to Scale

**Preparation:**
1. Document optimal parameters (chunk size, top_k, prompts)
2. Estimate costs: 566 reports × $0.05 = ~$28 API costs
3. Set up batch processing (50 reports at a time)
4. Plan for incremental validation

**Execution:**
1. Process all 4 textbooks (instead of 2)
2. Process all 566 KSP reports in batches
3. Periodic quality checks (random sample validation)
4. Comprehensive analysis after completion

**Timeline:** 2-3 weeks for full extraction + analysis

### If F1 = 0.60-0.69: Refinement Needed

**Focus areas:**
1. Prompt engineering (add few-shot examples)
2. Chunking optimization (test 256, 512, 768, 1024)
3. Retrieval tuning (test top_k = 3, 5, 10, 15)
4. Error analysis (categorize failure types)

**Timeline:** 1-2 weeks refinement, then re-pilot

### If F1 < 0.60: Major Revision Needed

**Consider:**
1. Different LLM (try Claude Opus 4.5 - more accurate)
2. Different embedding model (try larger model)
3. Manual annotation guidance for LLM
4. Hybrid approach (LLM + manual validation)

---

## Cost Summary

### Pilot Study (4 Reports + 2 Textbooks)

| Item | Cost | Notes |
|------|------|-------|
| Anthropic API | ~$0.22 | 16 API calls × $0.014 avg |
| Google Colab | $0 | Free tier sufficient |
| ChromaDB | $0 | Open source |
| Embedding Model | $0 | sentence-transformers (local) |
| **Total** | **~$0.22** | |

### Full Scale (566 Reports + 4 Textbooks)

| Item | Cost | Notes |
|------|------|-------|
| Anthropic API | ~$28 | 566 reports × $0.05 avg |
| Google Colab Pro | $10/mo | If exceeding free tier limits |
| ChromaDB | $0 | Open source |
| Embedding Model | $0 | sentence-transformers (local) |
| **Total** | **~$38-48** | One-time research cost |

**Note:** These are estimates. Actual costs depend on context length and output verbosity.

---

## Next Steps After Pilot

### Immediate (Within 1 Week)

1. **Review Results**
   - Examine `extracted_policies.json`
   - Check visualizations
   - Identify any systematic errors

2. **Calculate Metrics**
   - If gold standard: review F1 score
   - Manual quality check on random 20 policies
   - Document error patterns

3. **Decision Point**
   - F1 ≥ 0.70? → Prepare for scaling
   - F1 = 0.60-0.69? → Refinement iteration
   - F1 < 0.60? → Methodology revision

### Short-term (1-2 Weeks)

1. **Optimize Pipeline**
   - Tune parameters based on pilot learnings
   - Refine extraction prompts
   - Test on additional sample reports

2. **Create Research Artifacts**
   - Draft methodology section for paper
   - Create visualizations for presentation
   - Prepare code repository for sharing

### Medium-term (1-2 Months)

1. **Scale to Full Dataset**
   - Process 566 reports systematically
   - Process all 4 textbooks
   - Comprehensive theory-practice analysis

2. **Advanced Analysis**
   - Temporal trends (20 years evolution)
   - Cross-country comparisons  
   - Sector-specific deep dives
   - Theory coverage gaps

3. **Publication**
   - Write methodology paper
   - Present at conference
   - Share open-source code/data

---

## Additional Resources

### Documentation Files Provided

1. **ksp_pilot_complete.py** - Complete implementation (ready to run)
2. **google_colab_guide.md** - Detailed Colab setup instructions
3. **chromadb_reference.md** - ChromaDB operations reference
4. **textbook_integration_guide.md** - Theory-practice linking methodology
5. **CLAUDE.md** - This file

### External Resources

**ChromaDB:**
- Docs: https://docs.trychroma.com/
- GitHub: https://github.com/chroma-core/chroma

**Anthropic Claude:**
- Docs: https://docs.anthropic.com/
- API Reference: https://docs.anthropic.com/claude/reference

**sentence-transformers:**
- Docs: https://www.sbert.net/
- Models: https://huggingface.co/sentence-transformers

### Research Papers

**RAG Methodology:**
- Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Gao et al. (2023). "Retrieval-Augmented Generation for LLMs: A Survey"

**Development Economics (Your Literature):**
- Mokyr (2002) on propositional vs. prescriptive knowledge
- Ohno (2023) on translative adaptation
- Perkins (2013), Todaro (2020) - your textbook sources

---

## Quick Reference Commands

### In Google Colab

```python
# Check ChromaDB status
print(ksp_store.get_stats())
print(textbook_store.get_stats())

# View sample policy
import json
with open(f"{config.results_dir}/extracted_policies.json") as f:
    policies = json.load(f)
print(json.dumps(policies[0], indent=2))

# Re-run extraction (without re-indexing)
extracted_policies = extract_all_policies()

# Check if Drive is mounted
import os
print(os.path.exists("/content/drive/MyDrive/KSP_Pilot"))

# List uploaded PDFs
print(os.listdir("/content/drive/MyDrive/KSP_Pilot/data/raw/ksp_reports"))
```

---

## Summary

**This is a complete, ready-to-use implementation** for your pilot study with:
- ✅ Dual-collection RAG (KSP + Textbooks)
- ✅ Theory-practice linking
- ✅ Structured extraction with validation
- ✅ Automatic evaluation
- ✅ Comprehensive visualization
- ✅ Google Drive persistence

**To get started:**
1. Open Google Colab
2. Copy `ksp_pilot_complete.py` into a cell
3. Upload 4 KSP PDFs + 2 textbook PDFs
4. Add API key to Colab secrets
5. Run the cell

**Expected time:** 30-45 minutes total
**Expected cost:** ~$0.22
**Expected output:** 30-50 policies with theory links

**Your research contribution:** First systematic theory-practice linking in development economics at scale.

Good luck with your pilot study!

**Requirements:**
- Select 5 diverse reports for manual annotation
- Define annotation guidelines
- Have 2 annotators independently extract all policies
- Calculate inter-annotator agreement (Cohen's kappa or Fleiss' kappa)
- Resolve disagreements through discussion
- Save as JSON with same schema as extraction output

**Format:**
```json
{
  "report_id": "2015_VNM_Industrial_Policy",
  "annotator": "Expert_1",
  "policies": [
    {
      "policy_name": "Heavy and Chemical Industry Drive",
      "year_initiated": 1973,
      "organization": "Ministry of Trade and Industry",
      "challenge_addressed": "Transition from light to heavy industry",
      "policy_instruments": ["Tax incentives", "Credit allocation", "Infrastructure investment"],
      "sector": "Manufacturing",
      "development_stage": "middle_income",
      "evidence_quote": "In 1973, the Korean government launched the Heavy and Chemical Industry Drive...",
      "source_page": 45
    }
  ]
}
```

---

#### Task 2.2: Evaluation Metrics
**File:** `src/evaluation/metrics.py`

**Requirements:**
- Implement precision, recall, F1 for entity extraction
- Implement attribute-level accuracy (year, organization, sector)
- Compare predictions to gold standard
- Generate detailed evaluation report

**Key Functions:**
```python
class ExtractionEvaluator:
    def __init__(self, gold_standard_path: str)
    def evaluate_report(self, predictions: List[Dict], report_id: str) -> Dict
    def evaluate_all(self, predictions_path: str) -> pd.DataFrame
    def generate_report(self, output_path: str)
```

**Metrics:**
- Entity-level: Precision, Recall, F1 (does policy exist?)
- Attribute-level: Accuracy per field (year, org, sector, etc.)
- Error analysis: Categorize failure types

---

#### Task 2.3: Prompt Engineering
**File:** `src/extraction/prompts.py`

**Requirements:**
- Create multiple prompt variants (3-5 versions)
- Test on development set (not gold standard)
- Measure which prompts produce best structured output
- Document prompt evolution and rationale

**Prompt Templates:**
```python
EXTRACTION_PROMPTS = {
    "v1_basic": "...",
    "v2_few_shot": "...",  # Include 1-2 examples
    "v3_chain_of_thought": "...",  # Ask for reasoning
    "v4_structured": "...",  # Emphasize JSON schema
}
```

**Test:** Compare prompt versions on 10 sample extractions

---

### Phase 3: End-to-End Pipeline (Week 3)

#### Task 3.1: Orchestration Pipeline
**File:** `src/pipeline.py`

**Requirements:**
- Combine all components into single pipeline
- Process reports end-to-end (PDF → Extractions)
- Add progress tracking and logging
- Save intermediate outputs for debugging
- Support resuming from checkpoint

**Key Functions:**
```python
class KSPExtractionPipeline:
    def __init__(self, config_path: str)
    def process_report(self, pdf_path: str) -> List[Dict]
    def process_batch(self, pdf_paths: List[str], output_path: str)
    def evaluate_against_gold_standard(self)
```

**Test:** Run on 5-10 pilot reports

---

#### Task 3.2: Analysis Notebooks
**Files:** `notebooks/*.ipynb`

**Requirements:**

**01_data_exploration.ipynb:**
- Load and examine PDF structure
- Visualize section distribution
- Identify challenges (tables, formatting, etc.)

**02_prompt_engineering.ipynb:**
- Test different prompts
- Compare outputs
- Measure structured output success rate

**03_results_analysis.ipynb:**
- Load extraction results
- Compare to gold standard
- Visualize precision/recall
- Error analysis by failure type

**04_visualization.ipynb:**
- Embed extracted policies
- UMAP/t-SNE visualization
- Cluster analysis
- Compare sectoral vs. emergent categorizations

---

## Extraction Prompt Template

```python
EXTRACTION_PROMPT = """You are analyzing a Korean development policy document to extract structured information about policies, programs, and initiatives.

CONTEXT FROM DOCUMENT:
{retrieved_context}

TASK:
Extract ALL policies, programs, or initiatives mentioned in the context above.

For EACH policy/program, provide:
1. policy_name: Official title or clear description
2. year_initiated: Year it started (null if not mentioned)
3. organization: Responsible government ministry/agency (null if not mentioned)
4. challenge_addressed: What development problem did it address?
5. policy_instruments: List of specific mechanisms/tools used
6. sector: Economic sector (manufacturing, finance, agriculture, infrastructure, etc.)
7. development_stage: "early_industrialization", "middle_income", or "advanced"
8. evidence_quote: Direct quote from document that supports this (REQUIRED)

OUTPUT FORMAT:
Return a JSON array. Each element must follow this schema:
[
  {{
    "policy_name": "string",
    "year_initiated": integer or null,
    "organization": "string" or null,
    "challenge_addressed": "string",
    "policy_instruments": ["string", "string"],
    "sector": "string",
    "development_stage": "early_industrialization" | "middle_income" | "advanced",
    "evidence_quote": "string"
  }}
]

CRITICAL RULES:
- Only extract information explicitly stated in the context
- If information is not mentioned, use null
- Every extraction MUST include an evidence_quote
- Extract ALL policies mentioned, not just major ones
- Return valid JSON only (no markdown, no preamble, no explanation)
- If no policies found, return empty array: []

Begin extraction:
"""
```

## Expected Outputs

After completing the pilot, you should have:

### 1. Extracted Knowledge Base
**File:** `data/results/extracted_policies.json`
```json
[
  {
    "policy_name": "Heavy and Chemical Industry Drive",
    "year_initiated": 1973,
    "organization": "Ministry of Trade and Industry",
    "challenge_addressed": "Transition from light to heavy industry",
    "policy_instruments": ["Tax incentives", "Credit allocation"],
    "sector": "Manufacturing",
    "development_stage": "middle_income",
    "evidence_quote": "In 1973...",
    "source_report": "2015_VNM_Industrial_Policy",
    "report_year": "2015",
    "report_country": "VNM"
  }
  // ... more policies
]
```

### 2. Evaluation Report
**File:** `data/results/evaluation_report.json`
```json
{
  "overall_metrics": {
    "precision": 0.85,
    "recall": 0.78,
    "f1": 0.81,
    "num_reports": 5,
    "num_policies_extracted": 47,
    "num_policies_gold": 52
  },
  "by_report": [
    {
      "report_id": "2015_VNM_Industrial_Policy",
      "precision": 0.90,
      "recall": 0.80,
      "f1": 0.85
    }
  ],
  "attribute_accuracy": {
    "year_initiated": 0.72,
    "organization": 0.85,
    "sector": 0.91
  },
  "error_analysis": {
    "missing_year": 12,
    "hallucinated_policy": 3,
    "incorrect_organization": 5
  }
}
```

### 3. Visualizations
- UMAP plot of policy embeddings colored by sector
- Cluster analysis showing emergent categorizations
- Precision-recall curves
- Error distribution charts

## Success Criteria

**Minimum Viable Results for Pilot:**
- ✅ Process 5-10 reports successfully
- ✅ Extract structured JSON for all reports
- ✅ F1 score ≥ 0.70 on gold standard test set
- ✅ Identify and categorize major error types
- ✅ Document lessons learned for scaling

**Ideal Results:**
- ✅ F1 score ≥ 0.80
- ✅ <5% hallucination rate (policies not in document)
- ✅ Clear understanding of which fields are hardest to extract
- ✅ Optimized prompts ready for production

## Common Challenges & Solutions

### Challenge 1: PDF Quality Issues
**Problem:** Some PDFs have poor OCR, scanned images, complex tables
**Solution:**
- Use multiple extraction libraries (PyMuPDF + pdfplumber)
- Manual inspection of problematic files
- Document extraction failures systematically
- Consider OCR preprocessing if needed (Tesseract)

### Challenge 2: Ambiguous Policy Mentions
**Problem:** Policies mentioned indirectly or in passing
**Solution:**
- Design prompts to distinguish explicit vs. implicit mentions
- Add confidence scores to extractions
- Flag low-confidence extractions for manual review
- Document annotation guidelines clearly

### Challenge 3: LLM Hallucinations
**Problem:** LLM generates plausible but false information
**Solution:**
- Require evidence_quote for every extraction
- Implement faithfulness checking (verify quote exists in source)
- Use low temperature (0.1) for factual extraction
- Cross-reference with retrieved chunks

### Challenge 4: Temporal Ambiguity
**Problem:** Reports discuss policies from multiple time periods
**Solution:**
- Extract both mention_year (when discussed in report) and initiation_year
- Store temporal context from surrounding text
- Flag ambiguous dates for manual verification

### Challenge 5: JSON Parsing Failures
**Problem:** LLM doesn't always produce valid JSON
**Solution:**
- Use structured output parsing with error recovery
- Implement retry logic with clarifying prompts
- Log all parsing failures for analysis
- Consider using JSON mode in LLM if available

## Testing Strategy

### Unit Tests
```bash
# Test individual components
pytest tests/test_pdf_extractor.py
pytest tests/test_chunker.py
pytest tests/test_vector_store.py
pytest tests/test_extractor.py
```

### Integration Tests
```bash
# Test end-to-end on sample data
pytest tests/test_pipeline.py
```

### Validation Tests
```bash
# Compare to gold standard
python -m src.evaluation.metrics --gold-standard data/gold_standard/ --predictions data/results/extracted_policies.json
```

## Logging & Debugging

Enable comprehensive logging:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/extraction.log'),
        logging.StreamHandler()
    ]
)
```

**What to log:**
- PDF processing errors
- Chunking statistics (avg chunks per doc, size distribution)
- Retrieval queries and results
- LLM prompts and responses
- Extraction successes/failures
- Evaluation metrics

## Next Steps After Pilot

Once pilot is successful (F1 ≥ 0.70):

1. **Scale Up:**
   - Process all 566 reports
   - Use cloud GPU for faster processing (AWS/GCP)
   - Implement parallel processing

2. **Refine:**
   - Fine-tune prompts based on pilot errors
   - Adjust chunk size if needed
   - Consider using larger model (70B)

3. **Analyze:**
   - Cluster analysis of all extracted policies
   - Temporal analysis (evolution over 20 years)
   - Cross-country comparisons
   - Sector-specific patterns

4. **Publish:**
   - Write methodology paper
   - Share code and data repository
   - Present at conferences

## Resources & References

**Technical Documentation:**
- ChromaDB: https://docs.trychroma.com/
- LangChain: https://python.langchain.com/docs/get_started/introduction
- Ollama: https://github.com/ollama/ollama
- sentence-transformers: https://www.sbert.net/

**Research Papers:**
- Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
- Gao et al. (2023). "Retrieval-Augmented Generation for Large Language Models: A Survey"

**Your Literature Review:**
- Mokyr (2002) on propositional vs. prescriptive knowledge
- Ohno (2023) on translative adaptation and local learning
- World Bank (1998) on knowledge for development

## Contact & Support

**Primary Researcher:** [Your Name]
**Supervisor:** [Supervisor Name]
**Institution:** [Your University]

**Code Repository:** [GitHub URL once created]
**Questions:** [Your Email]

---

**Last Updated:** 2026-02-08
**Status:** Complete Implementation Ready
**Implementation:** `ksp_pilot_complete.py` (ready to run in Google Colab)
**Next Milestone:** Complete pilot study (4 reports + 2 textbooks) in 30-45 minutes