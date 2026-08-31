# Decisions

## Day 1 — Tokenizer decision

Checkpoint/tokenizer:
Local WordPiece demonstration tokenizer; mBERT comparison was also run using
`google-bert/bert-base-multilingual-cased`.

Corpus slice:
Synthetic samples (5).

Arabic fertility [MEASURED]:
1.08 mean fertility.

English fertility [MEASURED]:
1.14 mean fertility.

Truncation rate at max_length=10 [MEASURED]:
0%.

Known limitation:
The local WordPiece vocabulary is a small demonstration vocabulary and is not
representative of a production tokenizer. Unknown tokens can therefore occur.
The mBERT tokenizer comparison was performed on a small example sentence.

Decision and reason:
Use a checkpoint-matched pretrained tokenizer for model-facing processing.
The local WordPiece tokenizer was retained for demonstrating tokenisation
mechanics and measuring fertility/truncation in the Core lab. The mBERT
comparison showed Arabic subword segmentation such as `مع` + `##الجة` and
`م` + `##فيد` + `##ة`, highlighting the importance of measuring Arabic
tokenisation behavior rather than assuming word-level tokenisation.
# Applied NLP Pipeline & Architecture Documentation
**Program:** Bayan Applied Natural Language Processing Program  
**Organization:** Saudi Data & Artificial Intelligence Authority (SDAIA)  
**Track:** Multilingual NLP Engineering (Arabic & English)

---

## 1. Executive Summary & Objective
This project establishes an end-to-end, reproducible Natural Language Processing engineering pipeline across four foundational modules. It spans low-level text pre-processing and tokenisation mechanics, mathematical implementations of transformer attention layers, fine-tuning multilingual masked language models for text classification, and token-level classification tasks including Named Entity Recognition (NER) and Extractive Question Answering (QA).

All implementations enforce rigorous data contracts, group isolation to eliminate data leakage, and strict evaluation metrics.

---

## 2. Technical Architecture & Module Specifications

### Module 1: Text Processing, Sanitization & Subword Tokenisation
* **Unicode Normalization & Inspection:**
  * Strict Unicode normalization (`NFC`) to resolve visual ambiguities across Arabic glyphs and Latin characters[cite: 3].
  * Code point verification down to character-level metadata (e.g., verifying Alef with Hamza Above `U+0623`)[cite: 3].
* **Two-Copy Immutable Data Contract:**
  * `raw_text`: Immutable original text preserved for audit trails, compliance, and re-processing[cite: 3].
  * `model_text`: Sanitized and tokeniser-ready payload[cite: 3].
* **PII Redaction & Sanitization:**
  * Regex-based masking for Personally Identifiable Information (PII), converting email addresses to `<EMAIL>` and Saudi mobile numbers (`+966/05xxxxxxxx`) to `<PHONE>`[cite: 3].
  * HTML tag stripping, Tatweel/Kashida removal, and optional diacritics/Alef normalization[cite: 3].
* **Sentence Boundary Detection:**
  * Multilingual spaCy pipeline integration (`sentencizer`) with boundary rule adjustments for domain-specific abbreviations[cite: 3].
* **Vocabulary & Tokenisation Diagnostics:**
  * Implementation of custom `WordPiece` tokenisation with dedicated special tokens (`[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`)[cite: 3].
  * Diagnostic tracking via **Token Fertility** (subword ratio per lexical word) and **Truncation Rate** at fixed sequence bounds[cite: 3].

---

### Module 2: Attention Mechanics & Transformer Deep Dive
* **Scaled Dot-Product Attention:**
  * Vectorized NumPy implementation of the attention equation:
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  * Entropy measurement across unscaled vs. scaled logits to validate variance stabilization[cite: 3].
* **Masking Semantics:**
  * Causal and padding masking ensuring strictly zero weight attribution ($-\infty$ pre-softmax) across restricted indices and guaranteed row-stochastic property ($\sum \text{weights} = 1.0$)[cite: 3].
* **Multi-Head Attention (MHA) Projections:**
  * Tensor reshaping and transposition pipeline: `(batch, seq_len, d_model)` $\rightarrow$ `(batch, heads, seq_len, d_head)` $\rightarrow$ `(batch, seq_len, d_model)`[cite: 3].
  * Mathematical parity verification against native PyTorch `F.scaled_dot_product_attention` ($\Delta < 10^{-15}$)[cite: 3].
* **Transformer Parameter & Architecture Auditing:**
  * Analytical parameter audit comparing `google-bert/bert-base-multilingual-cased` (177.8M parameters, 51.8% embedding layer footprint) vs. `CAMeL-Lab/bert-base-arabic-camelbert-da` (109.1M parameters, 21.5% embedding footprint)[cite: 3].
  * Forward pass attention extraction on `distilbert-base-multilingual-cased`[cite: 3].

---

### Module 3: Multilingual Sequence Classification
* **Group-Isolated Dataset Partitioning:**
  * Strict `group_id` containment across `train`, `validation`, and `test` sets to guarantee `group_overlap = 0.0` (zero data leakage)[cite: 1, 2].
* **Baseline Benchmarking:**
  * Linear Support Vector Classifier (`LinearSVC`) fitted over character n-gram TF-IDF representations (`analyzer='char_wb'`, n-gram range `(3, 5)`)[cite: 1, 2].
* **Transformer Fine-Tuning Pipeline:**
  * Backbone: `distilbert/distilbert-base-multilingual-cased` with classification head[cite: 1, 2].
  * Strategy: Parameter freezing for CPU execution (partial fine-tuning of final transformer layer + classification head) using `AdamW`[cite: 1, 2].
  * Metric: Validation Macro-F1 checkpoint selection[cite: 1, 2].

---

### Module 4: Token Classification (NER) & Extractive Question Answering
* **Named Entity Recognition (NER):**
  * BIO tagging scheme across domain entities (`B-SERVICE`, `B-LOCATION`, `B-DATE`, `B-REF_NUM`, `B-ORG`, etc.)[cite: 1, 2].
  * Subword alignment contract using `word_ids()` mapped to `-100` for truncated/subsequent word pieces[cite: 1, 2].
  * Strict exact-boundary entity evaluation (Precision, Recall, Span F1)[cite: 1, 2].
* **Extractive Question Answering:**
  * Character-to-token offset mapping (`return_offsets_mapping=True`) handling context window clipping[cite: 1, 2].
  * Logit span post-processing with dynamic `null_threshold` scoring to support unanswerable/out-of-context queries[cite: 1, 2].

---

## 3. Environment & Runtime Specifications
* **Runtime:** Python 3.13.x[cite: 1, 3]
* **Pinned Dependencies:**
  * `transformers==5.15.1`[cite: 1, 2, 3]
  * `tokenizers==0.22.2`[cite: 1, 2, 3]
  * `scikit-learn==1.9.0`[cite: 1, 2]
  * `spacy==3.8.7`[cite: 3]
  * `torch` (PyTorch)[cite: 1, 2]
* **Deterministic Seed:** `42` across NumPy, PyTorch, and random generators[cite: 1, 2, 3]
