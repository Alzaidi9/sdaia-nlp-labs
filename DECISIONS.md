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
