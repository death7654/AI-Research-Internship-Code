# Gender Bias Detection in News Headlines

Research project completed at the **Amrita School of Artificial Intelligence, Amrita Vishwa Vidyapeetham** (May–June 2026) under the supervision of **Dr. Premjith B**, investigating automated detection of gender bias in news headlines using the [BABE dataset](https://www.kaggle.com/datasets/timospinde/babe-media-bias-annotations-by-experts).

**94% cross-validation accuracy** with `facebook/bart-large-mnli` embeddings, outperforming every 1.5B parameter model tested and a zero-shot LLM prompting pipeline.

---

## Overview

The project explored text classification for bias detection across two parallel tracks: a deep learning classifier trained on dense sentence embeddings, and a self-correcting LLM prompting pipeline using `SmolLM2-1.7B-Instruct`. The classifier approach was validated with 5-fold stratified cross-validation; the LLM pipeline was evaluated on a held-out test subset.

| Approach | Accuracy |
|---|---|
| TextBlob baseline | ~35% |
| Sentence-BERT + feed-forward classifier | ~88% |
| `facebook/bart-large-mnli` + feed-forward classifier (5-fold CV) | **~94%** |
| SmolLM2-1.7B-Instruct (zero-shot, self-correcting) | < BART classifier |

---

## Architecture

### Classifier

The core model is a feed-forward neural network that takes pre-computed sentence embeddings as input and outputs a biased/neutral classification.

```
Input (S-BERT embedding, 1024-dim)
  → Linear(1024) → ReLU → Dropout(0.3)
  → Linear(512)  → ReLU → Dropout(0.3)
  → Linear(256)  → ReLU → Dropout(0.3)
  → Linear(128)  → ReLU → Dropout(0.3)
  → Linear(embedding_dim)        ← enhanced embedding output
  → Linear(embedding_dim, 2)     ← classification head (biased / neutral)
```

### Loss Function

Training jointly optimizes cross-entropy and contrastive loss, with learnable scalar weights `c1` and `c2` optimized alongside the network via a separate Adam optimizer:

```
total_loss = exp(c1) * CE_loss + exp(c2) * contrastive_loss
```

The contrastive loss uses a `CombinedDataset` that dynamically samples positive (same-class) and negative (different-class) pairs per anchor at each epoch, computed via Euclidean distance with a margin of 1.0.

### Neutrosophic Fuzzy Scoring

A neutrosophic membership scoring function assigns three values — truth (T), indeterminacy (I), and falsity (F) — to each prediction by comparing the headline's embedding against the learned class centroids via cosine similarity. The final classification is determined by the maximum neutrosophic score `T - I - F` across classes. This produces soft, uncertainty-aware outputs rather than hard binary labels.

```python
def classify_article(memberships):
    scores = {label: (T - I - F) for label, (T, I, F) in memberships.items()}
    return max(scores, key=scores.get)
```

### Self-Correcting LLM Pipeline

A two-step prompting pipeline using `SmolLM2-1.7B-Instruct`:

1. **Step 1 — Initial classification**: few-shot prompt with reasoning chain, extracting `Classification: biased` or `Classification: neutral` from the output.
2. **Step 2 — Verification pass**: if Step 1 predicts `biased`, a second prompt cross-examines the headline, asking the model to reconsider whether it is "truly biased or presenting negative facts neutrally." The verification output overrides the initial prediction when it disagrees.

---

## Dataset

**BABE — Bias Annotations By Experts** ([Kaggle](https://www.kaggle.com/datasets/timospinde/babe-media-bias-annotations-by-experts))

US news headlines manually annotated as `biased` or `neutral` by domain experts.

**Data splits** (after deduplication on headline title):

| Split | Size |
|---|---|
| Train | ~80% |
| Validation | ~10% |
| Test | ~10% |

A key finding during development: an early train/validation leak (overlapping headline titles across splits) produced inflated validation accuracy. Fixing this required enforcing strict deduplication before splitting and verifying zero overlap with a leakage check:

```python
overlap = set(train_df['title']).intersection(set(val_df['title']))
assert len(overlap) == 0
```

---

## Results

Confusion matrix on the validation set with `facebook/bart-large-mnli` embeddings:

|  | Predicted Biased | Predicted Neutral |
|---|---|---|
| **True Biased** | 5259 | 37 |
| **True Neutral** | 386 | 8845 |

5259 out of 5296 biased headlines correctly identified (99.3% recall on the biased class). 386 neutral headlines were incorrectly flagged as biased (false positive rate ~4.2%).

---

## Key Finding

Representation quality was the primary bottleneck throughout the project, not model complexity. `facebook/bart-large-mnli` (400M parameters) outperformed every 1.5B parameter embedding model tested. The encoder swap alone — with no changes to the classifier architecture — was the single largest accuracy improvement in the pipeline.

---

## Stack

- **PyTorch** — model, training loop, contrastive loss
- **Sentence-Transformers** — `facebook/bart-large-mnli` encoder
- **Transformers** — `SmolLM2-1.7B-Instruct` for the LLM pipeline
- **Scikit-Learn** — `StratifiedKFold`, `classification_report`, baseline models
- **Pandas / NumPy** — data handling and preprocessing
- **KaggleHub** — dataset download

---

## Supervisor

**Dr. Premjith B**  
Assistant Professor (Senior Grade)  
Amrita School of Artificial Intelligence  
Amrita Vishwa Vidyapeetham, Coimbatore, India
