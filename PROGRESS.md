# Project Progress & Technical Benchmark Ledger

## Overall Test & Core Assertion Status
| Lab / Module | Technical Scope | Verification Assertions | Status |
| :--- | :--- | :---: | :---: |
| **Lab 1 — Part 1** | Text Preprocessing & Tokenisation Mechanics | `DAY1_NOTEBOOK1_CORE` | **PASS**[cite: 3] |
| **Lab 1 — Part 2** | Self-Attention & Transformer Architecture | `DAY1_NOTEBOOK2_CORE` | **PASS**[cite: 3] |
| **Lab 2 — Part 1** | Multilingual Sequence Classification | `DAY2_NOTEBOOK3_CORE` | **PASS**[cite: 1, 2] |
| **Lab 2 — Part 2** | Sequence Tagging (NER) & Extractive QA | `DAY2_NOTEBOOK4_CORE` | **PASS**[cite: 1, 2] |

---

## Verified Execution Benchmarks

### 1. Preprocessing & Tokenisation Metrics (Lab 1)
- [x] **Unicode Inspection:** Code point verification confirmed for `أ` (`U+0623`) and `é` (`U+00E9`)[cite: 3].
- [x] **Two-Copy Pipeline:** Raw text immutable; model text sanitized with `<EMAIL>` and `<PHONE>` masks[cite: 3].
- [x] **Sentence Segmentation:** Evaluated via spaCy pipeline across Arabic and English sentence pairs[cite: 3].
- [x] **Token Fertility Analysis:**
  - Arabic Mean Fertility: `1.08`[cite: 3]
  - English Mean Fertility: `1.14`[cite: 3]
  - Overall Mean Fertility: `1.36`[cite: 3]
  - Truncation Rate (@10 tokens): `0%`[cite: 3]
- [x] **Embedding Tensor Construction:** Input mapping verified with shape `(2, 12, 8)`[cite: 3].

### 2. Attention Mechanics & Transformer Audits (Lab 1)
- [x] **Entropy Scaling Effect:**
  - Unscaled Attention Entropy: `0.379`[cite: 3]
  - Scaled Attention Entropy ($\sqrt{d_k}$): `1.624`[cite: 3]
- [x] **Attention Constraints:** Causal masking verified with masked weights strictly zeroed (`w[0, 1] = 0.0`, `w[0, 2] = 0.0`)[cite: 3].
- [x] **Framework Parity:** Maximum numerical deviation between NumPy and PyTorch implementations: `4.441e-16`[cite: 3].
- [x] **Architecture Parameter Auditing:**
  - `bert-base-multilingual-cased`: `177,853,440` parameters (51.8% embedding footprint)[cite: 3]
  - `bert-base-arabic-camelbert-da`: `109,081,344` parameters (21.5% embedding footprint)[cite: 3]
- [x] **Live Transformer Forward Pass:** Parameter count verified at `134,734,080` on `distilbert-base-multilingual-cased`[cite: 3].

### 3. Classification Benchmark Ledger (Lab 2)
- [x] **Data Isolation Contract:** Split distribution: 24 train / 8 validation / 8 test across 20 distinct groups; `group_overlap = 0`[cite: 1, 2].
- [x] **Baseline Model (Char TF-IDF + LinearSVC):**
  - Validation Macro-F1: `0.6667`[cite: 1, 2]
  - Test Macro-F1: `0.7333`[cite: 1, 2]
- [x] **Fine-Tuned DistilBERT:**
  - Selected Best Epoch: `Epoch 9` (Mean Training Loss: `0.1160`)[cite: 1, 2]
  - Validation Macro-F1: `1.0000` ($\Delta = +0.3333$ over baseline)[cite: 1, 2]
  - Test Macro-F1: `0.8667`[cite: 1, 2]
  - Test Accuracy: `87.5%`[cite: 1, 2]

### 4. NER & Extractive QA Benchmarks (Lab 2)
- [x] **Token Classification Alignment:** Aligned subword tokens to `-100` padding mask[cite: 1, 2].
- [x] **NER Strict Entity Metrics:**
  - Precision: `0.6667`[cite: 1, 2]
  - Recall: `0.5000`[cite: 1, 2]
  - Span F1: `0.5714` (Gold: 4 entities, Predicted: 3 entities)[cite: 1, 2]
- [x] **Extractive QA Inference:**
  - Valid Span Extraction: `الرياض` (Score: `8.5`, Null Margin: `-8.5`)[cite: 1, 2]
  - Out-of-Context Null Rejection: Answer `None` (`no_answer_in_context`, Margin: `6.0`)[cite: 1, 2]
- [x] **Runtime Optimization:** Explicit GPU/CPU cache deallocation and garbage collection verified[cite: 1, 2].
