# KSP Knowledge Extraction - Pilot Study
# Complete Implementation for Google Colab
# 4 KSP Reports + 2 Development Economics Textbooks

"""
PILOT STUDY SETUP:
- 4 KSP advisory reports
- 2 development economics textbooks
- Dual-collection RAG with ChromaDB
- LLM extraction using Anthropic Claude API
- Full evaluation and visualization

GOOGLE COLAB USAGE:
- Runtime -> Change runtime type -> T4 GPU (for faster embeddings)
- Files will be saved to Google Drive for persistence
"""

# ============================================================================
# SECTION 1: SETUP & INSTALLATION
# ============================================================================

# Install required packages
!pip install -q pymupdf pdfplumber sentence-transformers chromadb anthropic pandas numpy scikit-learn matplotlib seaborn plotly networkx umap-learn python-dotenv

# Mount Google Drive for file persistence
from google.colab import drive
drive.mount('/content/drive')

# Create project directory in Google Drive
import os
project_dir = '/content/drive/MyDrive/KSP_Pilot'
os.makedirs(project_dir, exist_ok=True)
os.makedirs(f'{project_dir}/data/raw/ksp_reports', exist_ok=True)
os.makedirs(f'{project_dir}/data/raw/textbooks', exist_ok=True)
os.makedirs(f'{project_dir}/data/processed', exist_ok=True)
os.makedirs(f'{project_dir}/data/results', exist_ok=True)
os.makedirs(f'{project_dir}/data/gold_standard', exist_ok=True)
os.makedirs(f'{project_dir}/vector_db', exist_ok=True)

print("✓ Project directory created in Google Drive")
print(f"  Location: {project_dir}")
print("\nNext step: Upload your PDFs to:")
print(f"  - KSP reports: {project_dir}/data/raw/ksp_reports/")
print(f"  - Textbooks: {project_dir}/data/raw/textbooks/")

# ============================================================================
# SECTION 2: CONFIGURATION
# ============================================================================

import os
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

@dataclass
class Config:
    """Central configuration for the pilot study."""
    
    # Directories
    project_dir: str = project_dir
    ksp_dir: str = f"{project_dir}/data/raw/ksp_reports"
    textbook_dir: str = f"{project_dir}/data/raw/textbooks"
    processed_dir: str = f"{project_dir}/data/processed"
    results_dir: str = f"{project_dir}/data/results"
    vector_db_dir: str = f"{project_dir}/vector_db"
    
    # Chunking parameters
    ksp_chunk_size: int = 512
    textbook_chunk_size: int = 768
    chunk_overlap: int = 50
    
    # Embedding model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # LLM API
    llm_provider: str = "anthropic"  # or "openai"
    llm_model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.1
    max_tokens: int = 4000
    
    # Retrieval parameters
    ksp_top_k: int = 5
    textbook_top_k: int = 3
    
    # Collections
    ksp_collection: str = "ksp_reports_pilot"
    textbook_collection: str = "textbooks_pilot"

config = Config()

# API Key Setup
from google.colab import userdata

# Get API key from Colab secrets
# Go to: 🔑 icon in left sidebar -> Add new secret
# Name: ANTHROPIC_API_KEY, Value: your API key
try:
    ANTHROPIC_API_KEY = userdata.get('ANTHROPIC_API_KEY')
    print("✓ API key loaded from Colab secrets")
except:
    print("⚠ No API key found in Colab secrets")
    print("Please add ANTHROPIC_API_KEY in the secrets panel (🔑 icon)")
    ANTHROPIC_API_KEY = input("Or enter API key here: ")

# ============================================================================
# SECTION 3: PDF PROCESSING
# ============================================================================

import fitz  # PyMuPDF
import pdfplumber
import re
from pathlib import Path
from typing import List, Dict

class PDFProcessor:
    """Extract text from PDFs while preserving structure."""
    
    def __init__(self, pdf_path: str, source_type: str = "ksp"):
        """
        Args:
            pdf_path: Path to PDF file
            source_type: "ksp" or "textbook"
        """
        self.pdf_path = pdf_path
        self.source_type = source_type
        self.filename = Path(pdf_path).stem
        self.metadata = self._extract_metadata()
    
    def _extract_metadata(self) -> Dict:
        """Extract metadata from filename."""
        if self.source_type == "ksp":
            # Expected format: YYYY_CCC_Title.pdf
            # Example: 2015_VNM_Industrial_Policy.pdf
            pattern = r'(\d{4})_([A-Z]{3})_(.+)'
            match = re.match(pattern, self.filename)
            if match:
                return {
                    'source_type': 'ksp',
                    'year': match.group(1),
                    'country': match.group(2),
                    'title': match.group(3).replace('_', ' '),
                    'filename': self.filename
                }
        else:  # textbook
            # Expected format: Author_Year.pdf
            # Example: Perkins_2013.pdf
            return {
                'source_type': 'textbook',
                'filename': self.filename
            }
        
        return {'source_type': self.source_type, 'filename': self.filename}
    
    def extract_sections(self) -> List[Dict]:
        """Extract text organized by sections."""
        doc = fitz.open(self.pdf_path)
        sections = []
        current_section = {'title': 'Introduction', 'content': '', 'page': 1}
        
        for page_num, page in enumerate(doc, 1):
            # Extract text blocks with font information
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    text = ""
                    font_size = 0
                    
                    for span in line["spans"]:
                        text += span["text"]
                        font_size = max(font_size, span["size"])
                    
                    text = text.strip()
                    if not text:
                        continue
                    
                    # Detect section headers (larger font, shorter text)
                    if font_size > 12 and len(text) < 100:
                        # Save previous section
                        if current_section['content'].strip():
                            sections.append(current_section)
                        
                        # Start new section
                        current_section = {
                            'title': text,
                            'content': '',
                            'page': page_num
                        }
                    else:
                        current_section['content'] += text + "\n"
        
        # Add final section
        if current_section['content'].strip():
            sections.append(current_section)
        
        doc.close()
        return sections
    
    def extract_full_text(self) -> str:
        """Extract all text from PDF."""
        doc = fitz.open(self.pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    
    def get_page_count(self) -> int:
        """Get number of pages."""
        doc = fitz.open(self.pdf_path)
        count = len(doc)
        doc.close()
        return count

# Test PDF processing
print("Testing PDF processor...")
# Note: This will only work after you upload PDFs
# Uncomment and run after uploading files:

# test_pdf = f"{config.ksp_dir}/2015_VNM_Example.pdf"  # Replace with your actual file
# if os.path.exists(test_pdf):
#     processor = PDFProcessor(test_pdf, source_type="ksp")
#     print(f"✓ Metadata: {processor.metadata}")
#     sections = processor.extract_sections()
#     print(f"✓ Extracted {len(sections)} sections")
#     print(f"✓ Total pages: {processor.get_page_count()}")
# else:
#     print(f"Upload PDF to: {test_pdf}")

# ============================================================================
# SECTION 4: SEMANTIC CHUNKING
# ============================================================================

from langchain.text_splitter import RecursiveCharacterTextSplitter

class SemanticChunker:
    """Create semantic chunks from extracted text."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
    
    def chunk_sections(self, sections: List[Dict], metadata: Dict) -> List[Dict]:
        """Create chunks from sections with metadata."""
        chunks = []
        
        for section_idx, section in enumerate(sections):
            section_text = section['content']
            section_title = section['title']
            
            # Split section into smaller chunks if needed
            sub_chunks = self.splitter.split_text(section_text)
            
            for chunk_idx, chunk_text in enumerate(sub_chunks):
                chunk = {
                    'text': chunk_text,
                    'metadata': {
                        **metadata,  # Include all document metadata
                        'section_title': section_title,
                        'section_index': section_idx,
                        'section_page': section['page'],
                        'chunk_index': chunk_idx,
                        'chunk_id': f"{metadata['filename']}_s{section_idx}_c{chunk_idx}"
                    }
                }
                chunks.append(chunk)
        
        return chunks

# Test chunker
chunker_ksp = SemanticChunker(chunk_size=config.ksp_chunk_size)
chunker_textbook = SemanticChunker(chunk_size=config.textbook_chunk_size)
print("✓ Chunkers initialized")
print(f"  KSP chunk size: {config.ksp_chunk_size} characters")
print(f"  Textbook chunk size: {config.textbook_chunk_size} characters")

# ============================================================================
# SECTION 5: EMBEDDING & VECTOR DATABASE
# ============================================================================

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from tqdm import tqdm
import numpy as np

class VectorStore:
    """Manage embeddings and ChromaDB vector database."""
    
    def __init__(self, collection_name: str, persist_directory: str = None):
        """
        Args:
            collection_name: Name of the collection
            persist_directory: Where to save the database (Google Drive)
        """
        self.collection_name = collection_name
        
        # Initialize embedding model
        print(f"Loading embedding model: {config.embedding_model}...")
        self.embedding_model = SentenceTransformer(config.embedding_model)
        print("✓ Embedding model loaded")
        
        # Initialize ChromaDB with persistence
        if persist_directory is None:
            persist_directory = config.vector_db_dir
        
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✓ Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": f"Collection for {collection_name}"}
            )
            print(f"✓ Created new collection: {collection_name}")
    
    def add_documents(self, chunks: List[Dict], batch_size: int = 32):
        """Add document chunks to vector database."""
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [chunk['metadata']['chunk_id'] for chunk in chunks]
        
        print(f"Adding {len(documents)} documents to {self.collection_name}...")
        
        # Create embeddings in batches
        all_embeddings = []
        for i in tqdm(range(0, len(documents), batch_size), desc="Embedding"):
            batch_docs = documents[i:i+batch_size]
            embeddings = self.embedding_model.encode(
                batch_docs,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
            all_embeddings.extend(embeddings)
        
        # Add to ChromaDB in batches
        for i in tqdm(range(0, len(documents), batch_size), desc="Storing"):
            batch_end = min(i + batch_size, len(documents))
            
            self.collection.add(
                documents=documents[i:batch_end],
                embeddings=all_embeddings[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end]
            )
        
        print(f"✓ Added {len(documents)} chunks to collection")
    
    def search(self, 
               query: str, 
               n_results: int = 5,
               filter_dict: Dict = None) -> Dict:
        """Search for relevant chunks."""
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_dict
        )
        
        return results
    
    def get_stats(self) -> Dict:
        """Get collection statistics."""
        count = self.collection.count()
        return {
            'collection_name': self.collection_name,
            'total_chunks': count
        }

# Initialize vector stores
print("\n" + "="*60)
print("INITIALIZING VECTOR DATABASES")
print("="*60)

ksp_store = VectorStore(
    collection_name=config.ksp_collection,
    persist_directory=config.vector_db_dir
)

textbook_store = VectorStore(
    collection_name=config.textbook_collection,
    persist_directory=config.vector_db_dir
)

print("\n✓ Vector stores initialized")
print(f"  ChromaDB persisted to: {config.vector_db_dir}")

# ============================================================================
# SECTION 6: DOCUMENT INDEXING PIPELINE
# ============================================================================

def process_and_index_ksp_reports():
    """Process all KSP reports and add to vector database."""
    ksp_dir = Path(config.ksp_dir)
    pdf_files = list(ksp_dir.glob("*.pdf"))
    
    print(f"\nFound {len(pdf_files)} KSP reports")
    
    all_chunks = []
    
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        
        # Extract text
        processor = PDFProcessor(str(pdf_path), source_type="ksp")
        sections = processor.extract_sections()
        print(f"  Extracted {len(sections)} sections")
        
        # Create chunks
        chunks = chunker_ksp.chunk_sections(sections, processor.metadata)
        print(f"  Created {len(chunks)} chunks")
        
        all_chunks.extend(chunks)
    
    # Add to vector database
    if all_chunks:
        ksp_store.add_documents(all_chunks)
        
        # Save processed data
        output_path = f"{config.processed_dir}/ksp_chunks.json"
        with open(output_path, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        print(f"\n✓ Saved processed chunks to: {output_path}")
    
    return all_chunks

def process_and_index_textbooks():
    """Process textbooks and add to vector database."""
    textbook_dir = Path(config.textbook_dir)
    pdf_files = list(textbook_dir.glob("*.pdf"))
    
    print(f"\nFound {len(pdf_files)} textbooks")
    
    all_chunks = []
    
    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")
        
        # Extract text
        processor = PDFProcessor(str(pdf_path), source_type="textbook")
        sections = processor.extract_sections()
        print(f"  Extracted {len(sections)} sections")
        
        # Create chunks (larger for textbooks)
        chunks = chunker_textbook.chunk_sections(sections, processor.metadata)
        print(f"  Created {len(chunks)} chunks")
        
        all_chunks.extend(chunks)
    
    # Add to vector database
    if all_chunks:
        textbook_store.add_documents(all_chunks)
        
        # Save processed data
        output_path = f"{config.processed_dir}/textbook_chunks.json"
        with open(output_path, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        print(f"\n✓ Saved processed chunks to: {output_path}")
    
    return all_chunks

# ============================================================================
# SECTION 7: RUN DOCUMENT INDEXING
# ============================================================================

print("\n" + "="*60)
print("PHASE 1: DOCUMENT INDEXING")
print("="*60)

# Check if PDFs are uploaded
ksp_pdfs = list(Path(config.ksp_dir).glob("*.pdf"))
textbook_pdfs = list(Path(config.textbook_dir).glob("*.pdf"))

if len(ksp_pdfs) == 0:
    print(f"\n⚠ No KSP reports found in: {config.ksp_dir}")
    print("Please upload 4 KSP PDF reports to this directory")
    print("\nTo upload:")
    print("1. Click the folder icon in the left sidebar")
    print(f"2. Navigate to: {config.ksp_dir}")
    print("3. Right-click -> Upload")
else:
    print(f"\n✓ Found {len(ksp_pdfs)} KSP reports")
    ksp_chunks = process_and_index_ksp_reports()

if len(textbook_pdfs) == 0:
    print(f"\n⚠ No textbooks found in: {config.textbook_dir}")
    print("Please upload 2 textbook PDFs to this directory")
else:
    print(f"\n✓ Found {len(textbook_pdfs)} textbooks")
    textbook_chunks = process_and_index_textbooks()

# Print statistics
print("\n" + "="*60)
print("INDEXING COMPLETE")
print("="*60)
print(f"\nKSP Collection: {ksp_store.get_stats()}")
print(f"Textbook Collection: {textbook_store.get_stats()}")

# ============================================================================
# SECTION 8: LLM EXTRACTION
# ============================================================================

import anthropic
from typing import List, Dict
import json

class PolicyExtractor:
    """Extract structured policy information using Claude API."""
    
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = config.llm_model
        
        self.extraction_prompt = """You are analyzing development policy documents to extract structured information.

CONTEXT FROM KSP REPORT:
{ksp_context}

RELATED THEORETICAL CONCEPTS (from textbooks):
{theory_context}

TASK:
Extract ALL policies, programs, or initiatives mentioned in the KSP context.

For EACH policy/program, provide:
1. policy_name: Official title or clear description
2. year_initiated: Year it started (null if not mentioned)
3. organization: Responsible government ministry/agency (null if not mentioned)
4. challenge_addressed: What development problem did it address?
5. policy_instruments: List of specific mechanisms/tools used (e.g., ["Tax incentives", "Credit allocation"])
6. sector: Economic sector (e.g., "Manufacturing", "Finance", "Agriculture")
7. development_stage: "early_industrialization", "middle_income", or "advanced" (null if unclear)
8. evidence_quote: Direct quote from KSP document supporting this (REQUIRED - must be verbatim from context)
9. related_theory: Which theoretical concept from the textbook context relates to this policy? (null if none)

OUTPUT FORMAT:
Return ONLY a valid JSON array. No markdown, no preamble, no explanation.

[
  {{
    "policy_name": "string",
    "year_initiated": integer or null,
    "organization": "string" or null,
    "challenge_addressed": "string",
    "policy_instruments": ["string", "string"],
    "sector": "string",
    "development_stage": "early_industrialization" | "middle_income" | "advanced" | null,
    "evidence_quote": "string from KSP context",
    "related_theory": "string from textbook context" or null
  }}
]

CRITICAL RULES:
- Only extract information explicitly stated in the KSP context
- Every policy MUST have an evidence_quote from the KSP context
- If information not mentioned, use null
- Extract ALL policies mentioned, not just major ones
- Return valid JSON only
- If no policies found, return: []"""
    
    def extract_from_contexts(self, 
                              ksp_context: str, 
                              theory_context: str = "") -> List[Dict]:
        """Extract policies from KSP context with theory linking."""
        
        prompt = self.extraction_prompt.format(
            ksp_context=ksp_context,
            theory_context=theory_context if theory_context else "No theoretical context provided"
        )
        
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=config.max_tokens,
                temperature=config.temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            # Extract text from response
            content = response.content[0].text
            
            # Clean response (remove markdown if present)
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            content = content.strip()
            
            # Parse JSON
            policies = json.loads(content)
            
            if not isinstance(policies, list):
                policies = [policies]
            
            return policies
            
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response: {content[:200]}...")
            return []
        except Exception as e:
            print(f"Extraction error: {e}")
            return []

# Initialize extractor
extractor = PolicyExtractor(api_key=ANTHROPIC_API_KEY)
print("✓ Policy extractor initialized")

# ============================================================================
# SECTION 9: EXTRACTION PIPELINE
# ============================================================================

def extract_policies_from_report(report_filename: str) -> List[Dict]:
    """Extract policies from a single KSP report with theory linking."""
    
    print(f"\n{'='*60}")
    print(f"EXTRACTING FROM: {report_filename}")
    print('='*60)
    
    # Step 1: Query KSP collection for policy-relevant passages
    ksp_queries = [
        "Korean government policy program initiative implementation",
        "ministry organization agency institution",
        "economic development challenge problem solution",
        "policy instrument mechanism tool intervention"
    ]
    
    all_ksp_context = set()
    for query in ksp_queries:
        results = ksp_store.search(
            query=query,
            n_results=config.ksp_top_k,
            filter_dict={'filename': report_filename}
        )
        if results['documents'][0]:
            all_ksp_context.update(results['documents'][0])
    
    ksp_context = "\n\n---\n\n".join(all_ksp_context)
    print(f"Retrieved {len(all_ksp_context)} unique KSP chunks")
    
    # Step 2: Query textbook collection for related theory
    theory_query = f"development policy economic growth industrial policy"
    theory_results = textbook_store.search(
        query=theory_query,
        n_results=config.textbook_top_k
    )
    
    theory_context = "\n\n---\n\n".join(theory_results['documents'][0]) if theory_results['documents'][0] else ""
    print(f"Retrieved {len(theory_results['documents'][0])} textbook chunks")
    
    # Step 3: Extract policies with theory linking
    print("\nExtracting policies with Claude API...")
    policies = extractor.extract_from_contexts(ksp_context, theory_context)
    
    # Add source metadata
    for policy in policies:
        policy['source_report'] = report_filename
    
    print(f"✓ Extracted {len(policies)} policies")
    
    return policies

def extract_all_policies() -> List[Dict]:
    """Extract policies from all KSP reports."""
    ksp_pdfs = list(Path(config.ksp_dir).glob("*.pdf"))
    
    all_policies = []
    
    for pdf_path in ksp_pdfs:
        report_filename = pdf_path.stem
        policies = extract_policies_from_report(report_filename)
        all_policies.extend(policies)
    
    # Save results
    output_path = f"{config.results_dir}/extracted_policies.json"
    with open(output_path, 'w') as f:
        json.dump(all_policies, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print('='*60)
    print(f"Total policies extracted: {len(all_policies)}")
    print(f"Results saved to: {output_path}")
    
    return all_policies

# ============================================================================
# SECTION 10: RUN EXTRACTION
# ============================================================================

print("\n" + "="*60)
print("PHASE 2: POLICY EXTRACTION")
print("="*60)

# Check if we have indexed documents
if ksp_store.get_stats()['total_chunks'] == 0:
    print("\n⚠ No documents indexed yet!")
    print("Please run the indexing section (Section 7) first")
else:
    # Extract policies
    extracted_policies = extract_all_policies()
    
    # Display sample
    if extracted_policies:
        print("\nSample extracted policy:")
        print(json.dumps(extracted_policies[0], indent=2))

# ============================================================================
# SECTION 11: EVALUATION (if gold standard exists)
# ============================================================================

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

class Evaluator:
    """Evaluate extraction quality against gold standard."""
    
    def __init__(self, gold_standard_path: str):
        with open(gold_standard_path) as f:
            self.gold_standard = json.load(f)
    
    def evaluate(self, predictions: List[Dict]) -> Dict:
        """Evaluate predictions against gold standard."""
        
        # Group by report
        pred_by_report = {}
        for p in predictions:
            report = p.get('source_report', 'unknown')
            if report not in pred_by_report:
                pred_by_report[report] = []
            pred_by_report[report].append(p)
        
        gold_by_report = {}
        for g in self.gold_standard:
            report = g.get('source_report', 'unknown')
            if report not in gold_by_report:
                gold_by_report[report] = []
            gold_by_report[report].append(g)
        
        # Calculate metrics for each report
        results = []
        for report in set(list(pred_by_report.keys()) + list(gold_by_report.keys())):
            preds = pred_by_report.get(report, [])
            golds = gold_by_report.get(report, [])
            
            # Entity-level evaluation (policy names)
            pred_names = set(p['policy_name'].lower() for p in preds)
            gold_names = set(g['policy_name'].lower() for g in golds)
            
            tp = len(pred_names & gold_names)
            fp = len(pred_names - gold_names)
            fn = len(gold_names - pred_names)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'report': report,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'num_predicted': len(preds),
                'num_gold': len(golds)
            })
        
        # Calculate overall metrics
        total_tp = sum(r['tp'] for r in results)
        total_fp = sum(r['fp'] for r in results)
        total_fn = sum(r['fn'] for r in results)
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        return {
            'overall': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'total_predicted': sum(r['num_predicted'] for r in results),
                'total_gold': sum(r['num_gold'] for r in results)
            },
            'by_report': results
        }

# Check if gold standard exists
gold_standard_path = f"{config.project_dir}/data/gold_standard/annotations.json"
if os.path.exists(gold_standard_path):
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    evaluator = Evaluator(gold_standard_path)
    eval_results = evaluator.evaluate(extracted_policies)
    
    print("\nOverall Performance:")
    print(f"  Precision: {eval_results['overall']['precision']:.3f}")
    print(f"  Recall: {eval_results['overall']['recall']:.3f}")
    print(f"  F1 Score: {eval_results['overall']['f1']:.3f}")
    
    # Save evaluation results
    eval_output = f"{config.results_dir}/evaluation_results.json"
    with open(eval_output, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n✓ Evaluation results saved to: {eval_output}")
else:
    print(f"\n⚠ No gold standard found at: {gold_standard_path}")
    print("Create gold standard annotations to enable evaluation")

# ============================================================================
# SECTION 12: ANALYSIS & VISUALIZATION
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from collections import Counter

def analyze_policies(policies: List[Dict]):
    """Analyze extracted policies."""
    
    if not policies:
        print("No policies to analyze")
        return
    
    print("\n" + "="*60)
    print("POLICY ANALYSIS")
    print("="*60)
    
    df = pd.DataFrame(policies)
    
    # Basic statistics
    print(f"\nTotal policies extracted: {len(policies)}")
    print(f"Unique reports: {df['source_report'].nunique()}")
    
    # Sector distribution
    print("\nPolicies by Sector:")
    sector_counts = df['sector'].value_counts()
    print(sector_counts)
    
    # Development stage distribution
    if 'development_stage' in df.columns:
        print("\nPolicies by Development Stage:")
        stage_counts = df['development_stage'].value_counts()
        print(stage_counts)
    
    # Theory linking
    if 'related_theory' in df.columns:
        theory_linked = df['related_theory'].notna().sum()
        print(f"\nPolicies linked to theory: {theory_linked} ({theory_linked/len(df)*100:.1f}%)")
    
    # Visualization 1: Sector distribution
    plt.figure(figsize=(10, 6))
    sector_counts.plot(kind='barh')
    plt.title('Policies by Sector')
    plt.xlabel('Number of Policies')
    plt.tight_layout()
    plt.savefig(f"{config.results_dir}/sector_distribution.png", dpi=150)
    print(f"\n✓ Saved sector distribution chart")
    
    # Visualization 2: Theory-Practice Network
    if 'related_theory' in df.columns:
        create_theory_practice_network(policies)
    
    return df

def create_theory_practice_network(policies: List[Dict]):
    """Create network graph of theory-practice connections."""
    
    G = nx.Graph()
    
    for policy in policies:
        policy_name = policy['policy_name'][:30]  # Truncate for readability
        G.add_node(policy_name, node_type='policy', sector=policy.get('sector', 'Unknown'))
        
        if policy.get('related_theory'):
            theory = policy['related_theory'][:30]
            G.add_node(theory, node_type='theory')
            G.add_edge(policy_name, theory)
    
    # Visualize
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.5, iterations=50)
    
    # Color nodes by type
    colors = ['lightblue' if G.nodes[n].get('node_type') == 'policy' else 'lightcoral' for n in G.nodes()]
    
    nx.draw(G, pos, 
            node_color=colors,
            with_labels=True,
            font_size=8,
            node_size=500,
            alpha=0.7)
    
    plt.title('Theory-Practice Network\n(Blue=Policies, Red=Theories)')
    plt.tight_layout()
    plt.savefig(f"{config.results_dir}/theory_practice_network.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved theory-practice network")

# Run analysis
if 'extracted_policies' in locals() and extracted_policies:
    policy_df = analyze_policies(extracted_policies)

# ============================================================================
# SECTION 13: SUMMARY & NEXT STEPS
# ============================================================================

print("\n" + "="*60)
print("PILOT STUDY COMPLETE")
print("="*60)

print(f"\nResults saved in: {config.project_dir}")
print("\nKey outputs:")
print(f"  1. Processed chunks: {config.processed_dir}/")
print(f"  2. Vector database: {config.vector_db_dir}/")
print(f"  3. Extracted policies: {config.results_dir}/extracted_policies.json")
print(f"  4. Visualizations: {config.results_dir}/*.png")

if 'eval_results' in locals():
    print(f"\n  Overall F1 Score: {eval_results['overall']['f1']:.3f}")
    if eval_results['overall']['f1'] >= 0.70:
        print("  ✓ Pilot successful! Ready to scale to full 566 reports")
    else:
        print("  ⚠ Consider refining prompts or chunking strategy")

print("\nNext steps:")
print("  1. Review extracted policies for quality")
print("  2. Create gold standard annotations if not done")
print("  3. Refine extraction prompts based on errors")
print("  4. Scale to full dataset (566 reports)")

print("\n" + "="*60)