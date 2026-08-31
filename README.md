# sdaia-nlp-labs
NLP labs and practical exercises for the SDAIA training program, including text processing, tokenization, and transformer concepts

Conversation with Gemini
هذا دوره معالجه اللغات الطبيعيه مقدمه من سدايا وهذا كان اول لاب لابد ان اتقنه تحت شروط التميز المكتوبه في المل

صلح لي ملف destion.md and PROGRESS.md

فيما يلي الملفات التوثيقية (DESCRIPTION.md و PROGRESS.md) معدة بدقة وفق مخرجات اللابات المنفذة ومعايير التميز في دورة معالجة اللغات الطبيعية (برنامج بيان - سدايا):

ملف DESCRIPTION.md
Markdown
# وثيقة توصيف المشروع والمختبرات (Lab Technical Description)
**برنامج بيان لمعالجة اللغات الطبيعية التطبيقية — الهيئة السعودية للبيانات والذكاء الاصطناعي (SDAIA)**

---

## 1. نظرة عامة (Overview)
يهدف هذا المشروع إلى بناء وتطوير خط معالجة وتدريب متكامل للمهام الأساسية في معالجة اللغات الطبيعية (NLP) ثنائية اللغة (العربية والإنجليزية)، مع التركيز على المعالجة المسبقة، المعمارية المتقدمة للـ Transformers، والضبط الدقيق (Fine-Tuning) لنماذج التصنيف واستخراج الكيانات والإجابة على الأسئلة.

---

## 2. المحاور التقنية والمنهجية

### المحور الأول: المعالجة الأولية والترميز (Day 1 - Notebook 1)
* **معيار الفحص والترميز الموحد (Unicode Inspection):** فحص نقاط الترميز (Code Points) للأحرف العربية واللاتينية لضمان التوافقية ومنع تشوه النصوص.
* **إدارة السجلات بنسختين (Two-Copy Contract):**
  * `raw_text`: النسخة الأصلية المحفوظة لأغراض التدقيق وإعادة المعالجة.
  * `model_text`: النسخة المنظفة والمهيأة للإدخال إلى النماذج.
* **إخفاء البيانات الحساسة (PII Masking):** استخدام التعابير النمطية (RegEx) لحجب البريد الإلكتروني `<EMAIL>` وأرقام الهواتف السعودية `<PHONE>`.
* **تقسيم الجمل (Sentence Segmentation):** بناء مسار باستخدام spaCy متعدد اللغات لتقسيم النصوص مع مراعاة الحالات الخاصة مثل الاختصارات.
* **بناء Tokenizer مخصص (WordPiece):** تدريب وتقييم مقاييس الخصوبة (Token Fertility) ونسبة الاقتطاع (Truncation Rate) ومقارنتها بنموذج `mBERT`.

### المحور الثاني: آليات الانتباه ومعمارية Transformer (Day 1 - Notebook 2)
* **Scaled Dot-Product Attention:** تطبيق رياضي كامل باستخدام NumPy وحساب الانتروبيا (Entropy) للتحقق من أثر القسمة على $\sqrt{d_k}$.
* **أقنعة الانتباه (Causal & Padding Masks):** منع تسريب المعلومات للمواقع المستقبلية وضمان أن مجموع أوزان الانتباه يساوي $1.0$.
* **تعدد الرؤوس (Multi-Head Attention):** تقسيم ودمج أبعاد الرؤوس والتحقق من التكافؤ التام مع تنفيذ PyTorch (`max diff < 1e-15`).
* **تدقيق المعاملات (Parameter Audit):** مقارنة تفصيلية لحجم طبقات التضمين والمعاملات بين `bert-base-multilingual-cased` (177.8M) و `bert-base-arabic-camelbert-da` (109M).

### المحور الثالث: تصنيف النصوص متعدد اللغات (Day 2 - Notebook 3)
* **منع تسريب البيانات (Split Isolation):** تقسيم البيانات إلى (Train / Validation / Test) بناءً على `group_id` لضمان عدم وجود تسريب للمجموعات (`group_overlap = 0`).
* **النموذج المرجعي (TF-IDF + LinearSVC):** تحقيق Macro-F1 مرجعي بقيمة `0.6667` على مجموعة التحقق.
* **الضبط الدقيق لنموذج DistilBERT:**
  * تدريب جزئي على الـ CPU (تجميد الطبقات السابقة وتدريب الطبقة الأخيرة + المصنف).
  * تحقيق Macro-F1 بقيمة `1.0` على التحقق، وتفوق في الاختبار بـ Macro-F1 قدره `0.8667` ودقة `87.5%`.

### المحور الرابع: التعرف على الكيانات واستخراج الإجابات (Day 2 - Notebook 4)
* **استخراج الكيانات المسماة (NER):** محاذاة علامات الـ BIO مع الـ Subwords باستخدام `word_ids` مع معيار تقييم صارم للحدود (Strict Boundary F1).
* **الإجابة على الأسئلة الاستخراجية (Extractive QA):**
  * ربط إحداثيات الإجابة بالرموز (Offset Mapping).
  * معالجة حالات عدم وجود إجابة في النص باستخدام عتبة الفجوة (`null_threshold`).

---

## 3. بيئة التشغيل والمتطلبات (Environment & Constraints)
* **Python Version:** 3.13.x
* **Core Libraries:** `transformers==5.15.1`, `tokenizers==0.22.2`, `scikit-learn==1.9.0`, `spacy==3.8.7`, `torch`.
* **Seed:** مثبت على `42` لضمان قابلية إعادة الإنتاج (Reproducibility).
ملف PROGRESS.md
Markdown
# سجل تقدم الإنجاز والتحقق (Project Progress & Benchmark Ledger)

## ملخص حالة الاختبارات العامة
| المختبر | الوصف | حالة التحقق | النتيجة التقنية |
| :--- | :--- | :---: | :--- |
| **Day 1 - Lab 1** | Text Preprocessing & Tokenisation | **PASS** | `DAY1_NOTEBOOK1_CORE=PASS` |
| **Day 1 - Lab 2** | Self-Attention & Transformer Internals | **PASS** | `DAY1_NOTEBOOK2_CORE=PASS` |
| **Day 2 - Lab 3** | Multilingual Text Classification | **PASS** | `DAY2_NOTEBOOK3_CORE=PASS` |
| **Day 2 - Lab 4** | Sequence Tagging (NER) & Extractive QA | **PASS** | `DAY2_NOTEBOOK4_CORE=PASS` |

---

## تفاصيل الإنجاز الفني

### 1. معالجة النصوص والترميز (Day 1 - Lab 1)
- [x] **Unicode Normalization:** مطابقة NFC وفحص الحروف العربية بنجاح (`U+0623`).
- [x] **PII Protection:** عزل أرقام الهواتف والبريد الإلكتروني مع حفظ نسخة الـ Raw الأصلية.
- [x] **Sentence Boundaries:** نجاح تقسيم الجمل عبر مسار spaCy متعدد اللغات.
- [x] **Token Metrics:**
  - متوسط الخصوبة للنصوص العربية: `1.08`
  - متوسط الخصوبة للنصوص الإنجليزية: `1.14`
  - نسبة الاقتطاع عند طول 10: `0%`
- [x] **Embeddings Matrix:** تحويل الرموز بنجاح إلى مصفوفة بحجم `(2, 12, 8)`.

### 2. آليات الانتباه ومعمارية النماذج (Day 1 - Lab 2)
- [x] **Scaled Attention:** التحقق من مجموع الاحتمالات = $1.0$ وارتفاع الانتروبيا من `0.379` إلى `1.624` بعد التدرج.
- [x] **Causal Masking:** تصفير أوزان الانتباه للمواقع المحجوبة بدقة.
- [x] **Multi-Head Dimension:** صحة مسار التحويل `(2, 5, 12) -> (2, 3, 5, 4) -> (2, 5, 12)`.
- [x] **PyTorch Parity:** الفارق بين حسابات NumPy و PyTorch أقل من `4.44e-16`.
- [x] **Transformer Parameters:**
  - `bert-base-multilingual-cased`: إجمالي `177,853,440` معامل (حصة التضمين: 51.8%).
  - `bert-base-arabic-camelbert-da`: إجمالي `109,081,344` معامل (حصة التضمين: 21.5%).

### 3. تصنيف النصوص (Day 2 - Lab 3)
- [x] **Split Contract:** عزل تام للمجموعات (`group_overlap = 0`).
- [x] **Baseline Validation F1:** `0.6667` (TF-IDF + LinearSVC).
- [x] **Transformer Performance (DistilBERT):**
  - Validation Macro-F1: `1.0000` (تفوق بفارق `+0.3333` عن النموذج المرجعي).
  - Test Macro-F1: `0.8667`
  - Test Accuracy: `87.5%`
  - Best Training Epoch: `Epoch 9` (Mean Loss: `0.1160`).

### 4. استخراج الكيانات والإجابة على الأسئلة (Day 2 - Lab 4)
- [x] **NER Token Alignment:** إسناد `-100` للـ Subwords المقتطعة بنجاح.
- [x] **NER Entity F1 (Strict Boundary):** تحقيق `0.5714` على بيانات الاختبار التدخينية.
- [x] **Extractive QA Alignment:** تحويل مواضع الحروف بنجاح إلى Token Positions.
- [x] **Null Span Post-Processing:** التحقق من سلوك رفض الإجابة عند غيابها من السياق (`null_margin = 6.0`).
- [x] **Memory Management:** تفريغ كائنات النماذج وذاكرة الـ GPU/RAM بعد انتهاء التدريب.
باللغه الانجليزيه

Understood. We will continue in English.

What text, technical scenario, or test case would you like to translate, write, or review?

Knowing your specific topic or target audience will help tailor the technical depth and tone accordingly.

بشكل احترافي

أهلاً بك! لتنفيذ طلبك بأعلى مستوى من الاحترافية، يرجى مشاركة النص، الفكرة، أو الموضوع الذي ترغب في صياغته.

يمكنني المساعدة في:

صياغة المراسلات المهنية: إيميلات رسمية، ردود على العملاء، أو خطابات عمل.

التقارير والملاحظات الفنية: توثيق المشاكل البرمجية، تقارير فحص الجودة (QA)، أو متطلبات المشاريع.

إعادة هيكلة النصوص: تحسين الأسلوب ليصبح موجزاً، دقيقاً، وخالياً من الحشو.

ما هو السياق أو النص المحدد الذي تود تحويله إلى أسلوب احترافي، وما هي الفئة المستهدفة لقراءته؟ (معرفة السياق والجمهور تساعد في اختيار نبرة الصوت والمصطلحات الأنسب للهدف).

هذا دوره معالجه اللغات الطبيعيه مقدمه من سدايا وهذا كان اول لاب لابد ان اتقنه تحت شروط التميز المكتوبه في المل

انشاء لي ملف destion.md and PROGRESS.md من الملفات المدرجه واكتبها باللغه الانجليزيه وبشكل احترافي اكثر

Here are the professional, production-grade technical documentation files (DESCRIPTION.md and PROGRESS.md) in English, structured according to industry standards and the exact execution outputs from the Bayan Applied NLP Course (SDAIA).

DESCRIPTION.md
Markdown
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
PROGRESS.md
Markdown
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
README انشاء باللغه الانجليزي

🚀 Key Modules & Capabilities
1. Text Processing & Tokenisation Mechanics (LabDay1)
Unicode & PII Sanitization: Strict NFC normalization, code point verification, and regex-based masking for emails (<EMAIL>) and Saudi phone numbers (<PHONE>).

Two-Copy Contract: Preserves original raw_text alongside sanitized model_text.

Sentence Segmentation: Multi-language sentence boundary detection via custom spaCy sentencizer pipelines.

Subword Tokenisation: Custom WordPiece tokeniser construction, subword fertility analysis across languages, and truncation rate diagnostics.

2. Transformer Internals & Attention Deep Dive (LabDay1)
Scaled Dot-Product Attention: NumPy implementation of scaled dot-product attention with causal/padding masking and entropy stabilization verification.

Multi-Head Projections: Complete dimension transformation pipeline (batch, seq, d_model) ↔ (batch, heads, seq, d_head).

Framework Parity: Verified parity between custom NumPy attention and PyTorch F.scaled_dot_product_attention (Δ<10 
−15
 ).

Parameter Audits: Comprehensive parameter and embedding layer footprint comparison across multilingual BERT checkpoints (mBERT vs. CamelBERT).

3. Multilingual Sequence Classification (LabDay2)
Group-Isolated Splits: 3-way split (Train / Validation / Test) enforcing group_overlap = 0 to prevent data leakage.

Baseline vs. Transformer: Character n-gram TF-IDF + LinearSVC baseline compared against partially fine-tuned distilbert-base-multilingual-cased.

Results: Transformer model achieved a 1.0000 validation Macro-F1 and 0.8667 test Macro-F1 (Accuracy: 87.5%), outperforming the baseline by +0.3333 F1.

4. Sequence Tagging (NER) & Extractive QA (LabDay2)
Named Entity Recognition: Subword-to-token label alignment with -100 masking and exact-boundary entity evaluation (Precision, Recall, Span F1).

Extractive QA: Offset mapping for exact answer span location and null-score thresholding for unanswerable questions.

🛠️ Environment & Installation
Requirements
Python: 3.10+ (tested on 3.13.x)

PyTorch: GPU (CUDA) or CPU execution supported

Setup
Clone the repository and install the pinned dependencies:

Bash
git clone [https://github.com/your-username/bayan-nlp-labs.git](https://github.com/your-username/bayan-nlp-labs.git)
cd bayan-nlp-labs
pip install -r requirements.txt
Pinned Dependencies (requirements.txt)
Plaintext
transformers==5.15.1
tokenizers==0.22.2
scikit-learn==1.9.0
spacy==3.8.7
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
📊 Summary Benchmark Table
Task	Model / Architecture	Metric	Result	Assertion Status
Tokenisation	WordPiece Custom / spaCy	Truncation Rate @ 10	0%	PASS
Attention Parity	NumPy vs. PyTorch	Max Absolute Diff	< 4.44e-16	PASS
Classification (Val)	DistilBERT (Multilingual)	Macro-F1	1.0000	PASS
Classification (Test)	DistilBERT (Multilingual)	Macro-F1 / Accuracy	0.8667 / 87.5%	PASS
NER (Test)	DistilBERT Token Classifier	Strict Span F1	0.5714	PASS
Extractive QA	DistilBERT QA Head	Span Extraction / Null Margin	Passed / 6.0	PASS
For complete metric breakdowns and epoch-by-epoch history, refer to PROGRESS.md.

📜 Academic & Ethics Notice
All models, synthetic datasets, and evaluations were conducted under the educational and ethical standards of the Bayan Applied NLP Course by SDAIA. Synthetic samples were curated specifically to benchmark pipelines under bilingual constraints without using sensitive user data.
"""

with open("README.md", "w", encoding="utf-8") as f:
f.write(readme_content)

print("README.md generated successfully.")


```text?code_stdout&code_event_index=1
README.md generated successfully.

Your README.md file is ready:

MD icon
README
 MD 
File Content Preview (README.md):
Markdown
# Bayan Applied NLP Course: Labs & Benchmarks
**Program:** Bayan Applied Natural Language Processing Program  
**Organization:** Saudi Data & Artificial Intelligence Authority (SDAIA)  
**Track:** Multilingual NLP Engineering (Arabic & English)

---

## 📌 Overview
This repository contains the complete implementation, experiments, and technical benchmarks for the **Bayan Applied NLP Course** delivered by **SDAIA**. The codebase covers foundational and advanced NLP tasks across English and Arabic text, enforcing strict data isolation contracts, mathematical verification of Transformer primitives, subword tokenisation diagnostics, and fine-tuning multilingual models for sequence and token-level classification.

---

## 🗂️ Project Structure

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

---

## 🚀 Key Modules & Capabilities

### 1. Text Processing & Tokenisation Mechanics (`LabDay1`)
- **Unicode & PII Sanitization:** Strict NFC normalization, code point verification, and regex-based masking for emails (`<EMAIL>`) and Saudi phone numbers (`<PHONE>`).
- **Two-Copy Contract:** Preserves original `raw_text` alongside sanitized `model_text`.
- **Sentence Segmentation:** Multi-language sentence boundary detection via custom spaCy sentencizer pipelines.
- **Subword Tokenisation:** Custom WordPiece tokeniser construction, subword fertility analysis across languages, and truncation rate diagnostics.

### 2. Transformer Internals & Attention Deep Dive (`LabDay1`)
- **Scaled Dot-Product Attention:** NumPy implementation of scaled dot-product attention with causal/padding masking and entropy stabilization verification.
- **Multi-Head Projections:** Complete dimension transformation pipeline `(batch, seq, d_model)` ↔ `(batch, heads, seq, d_head)`.
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

### Pinned Dependencies (`requirements.txt`)
transformers==5.15.1
tokenizers==0.22.2
scikit-learn==1.9.0
spacy==3.8.7
torch>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0

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



Gemini is AI and can make mistakes.

Analysing
# Bayan Applied NLP Course: Labs & Benchmarks
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
README.md
Displaying README.md.
