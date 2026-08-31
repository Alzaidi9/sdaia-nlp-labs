Bayan Applied NLP Course: Labs & Benchmarks
**Program:** Bayan Applied Natural Language Processing Program  
**Organization:** Saudi Data & Artificial Intelligence Authority (SDAIA)  
**Track:** Multilingual NLP Engineering (Arabic & English)

---

## 📌 Overview
This repository contains the complete implementation, experiments, and technical benchmarks for the **Bayan Applied NLP Course** delivered by **SDAIA**. The codebase covers foundational and advanced NLP tasks across English and Arabic text, enforcing strict data isolation contracts, mathematical verification of Transformer primitives, subword tokenisation diagnostics, and fine-tuning multilingual models for sequence and token-level classification.

---

## 🗂️ Project Structure

```text
.
├── notebooks/
│   ├── LabDay1.ipynb          # Lab 1: Text Pre-processing, Tokenisation & Attention Primitives
│   ├── LabDay2.ipynb          # Lab 2: Sequence Classification, NER & Extractive QA
│   └── LabDay3.ipynb          # Lab 3: Retrieval, Semantic Search & Evaluation
├── docs/
│   ├── DESCRIPTION.md         # Detailed technical architecture & pipeline specifications
│   └── PROGRESS.md            # Benchmark ledger, assertions status & test results
├── data/                      # Dataset samples (classification, NER, QA)
├── requirements.txt           # Pinned dependencies for reproducible execution
└── README.md                  # Project overview and quickstart guide
```

---

## 🚀 Key Modules & Capabilities

### 1. Text Processing & Tokenisation Mechanics (`LabDay1`)
- **Unicode & PII Sanitization:** Strict NFC normalization, code point verification, and regex-based masking for emails (`<EMAIL>`) and Saudi phone numbers (`<PHONE>`).
- **Two-Copy Contract:** Preserves original `raw_text` alongside sanitized `model_text`.
- **Sentence Segmentation:** Multi-language sentence boundary detection via custom spaCy sentencizer pipelines.
- **Subword Tokenisation:** Custom WordPiece tokeniser construction, subword fertility analysis across languages, and truncation rate diagnostics.

### 2. Transformer Internals & Attention Deep Dive (`LabDay1`)
- **Scaled Dot-Product Attention:** NumPy implementation of scaled dot-product attention with causal/padding masking and entropy stabilization verification.
- **Multi-Head Projections:** Complete dimension transformation pipeline `(batch, seq, d_model)` $\leftrightarrow$ `(batch, heads, seq, d_head)`.
- **Framework Parity:** Verified parity between custom NumPy attention and PyTorch `F.scaled_dot_product_attention` ($\Delta < 10^{-15}$).
- **Parameter Audits:** Comprehensive parameter and embedding layer footprint comparison across multilingual BERT checkpoints (`mBERT` vs. `CamelBERT`).

### 3. Multilingual Sequence Classification (`LabDay2`)
- **Group-Isolated Splits:** 3-way split (Train / Validation / Test) enforcing `group_overlap = 0` to prevent data leakage.
- **Baseline vs. Transformer:** Character n-gram TF-IDF + LinearSVC baseline compared against partially fine-tuned `distilbert-base-multilingual-cased`.
- **Results:** Transformer model achieved a **1.0000** validation Macro-F1 and **0.8667** test Macro-F1 (Accuracy: **87.5%**), outperforming the baseline by **+0.3333** F1.

### 4. Sequence Tagging (NER) & Extractive QA (`LabDay2`)
- **Named Entity Recognition:** Subword-to-token label alignment with `-100` masking and exact-boundary entity evaluation (Precision, Recall, Span F1).
- **Extractive QA:** Offset mapping for exact answer span location and null-score thresholding for unanswerable questions.

---

## 🛠️ Environment & Installation

### Requirements
- **Python:** `3.10+` (tested on `3.13.x`)
- **PyTorch:** GPU (CUDA) or CPU execution supported

### Setup
Clone the repository and install the pinned dependencies:

```bash
git clone https://github.com/your-username/bayan-nlp-labs.git
cd bayan-nlp-labs
pip install -r requirements.txt
```

### Pinned Dependencies (`requirements.txt`)
```text
transformers==5.15.1
tokenizers==0.22.2
scikit-learn==1.9.0
spacy==3.8.7
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
```

---

## 📊 Summary Benchmark Table

| Task | Model / Architecture | Metric | Result | Assertion Status |
| :--- | :--- | :---: | :---: | :---: |
| **Tokenisation** | WordPiece Custom / spaCy | Truncation Rate @ 10 | `0%` | `PASS` |
| **Attention Parity** | NumPy vs. PyTorch | Max Absolute Diff | `< 4.44e-16` | `PASS` |
| **Classification (Val)** | DistilBERT (Multilingual) | Macro-F1 | `1.0000` | `PASS` |
| **Classification (Test)**| DistilBERT (Multilingual) | Macro-F1 / Accuracy | `0.8667` / `87.5%` | `PASS` |
| **NER (Test)** | DistilBERT Token Classifier | Strict Span F1 | `0.5714` | `PASS` |
| **Extractive QA** | DistilBERT QA Head | Span Extraction / Null Margin | `Passed / 6.0` | `PASS` |

*For complete metric breakdowns and epoch-by-epoch history, refer to [PROGRESS.md](PROGRESS.md).*

---

## 📜 Academic & Ethics Notice
All models, synthetic datasets, and evaluations were conducted under the educational and ethical standards of the **Bayan Applied NLP Course** by **SDAIA**. Synthetic samples were curated specifically to benchmark pipelines under bilingual constraints without using sensitive user data.
