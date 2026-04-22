# Localization Terminology Extractor

A Python-based terminology extraction pipeline for localization workflows. This tool automatically surfaces domain-specific term candidates from source text using three progressive filtering methods: LDA topic modeling, TF-IDF filtering, and Brown Corpus comparison.

## How It Works

```
Source Text (.txt)
  → LDA Topic Modeling       (broad candidate list)
  → TF-IDF Filtering         (keep statistically distinctive terms)
  → Brown Corpus Comparison  (remove common English words)
  → Terminology Candidate List
```

## Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# Install dependencies
pip install scikit-learn nltk
```

## Usage

Add your source text to `sample_corpus.txt`, then run:

```bash
# Step 1: LDA unfiltered candidate list
python3 lda_term_extractor.py

# Step 2: TF-IDF filtered list
python3 lda_term_extractor.py --filter tfidf

# Step 3: Brown Corpus comparison (recommended final output)
python3 lda_term_extractor.py --filter brown
```

## Example Output

**Corpus:** [What Makes a Translation Management System](https://phrase.com/blog/posts/translation-management-system-how-it-works/) — Phrase.com

**Brown Corpus filtered output:**
```
api, automatically, developers, integration, languages, localization,
managers, productivity, target, tasks, tms, translation, translators,
website, workflow ...
```

**After manual review — added back:**

| Term | Reason |
|---|---|
| `translation memory` | Core L10n concept; filtered because "memory" is common in Brown Corpus |
| `machine translation` | Core L10n technology; filtered because "machine" is a common word |
| `quality assurance` | Key localization workflow step |
| `source` | As in source language / source content — essential L10n concept |
| `review` | Defined stage in professional localization workflows |

**After manual review — removed:**

| Term | Reason |
|---|---|
| `allows` | Common verb, not a domain term |
| `faster` | General adjective, no terminological value |
| `updated` | Too generic |
| `spanish` | Incidental example in the article, not a localization term |

## Key Insight

Automated extraction always requires human review. Common issues include:

- **False positives** — general words that pass statistical filters (e.g. `allows`, `faster`)
- **Duplicate word forms** — `translation`, `translations`, and `translated` appear as separate entries; lemmatization should be applied before extraction to merge these
- **Missing multi-word terms** — single-word extraction misses compound terms like "translation memory" and "machine translation"

## Real-World Localization Workflow

This tool fits into a professional terminology management pipeline:

```
1. Extract candidates from source corpus   (this tool)
       ↓
2. Human review and SME validation
       ↓
3. Export to TBX format
       ↓
4. Import into TMS term base (SDL MultiTerm, memoQ, Phrase)
       ↓
5. Enforce terminology during translation via CAT tool
```

## Tech Stack

- Python 3.10+
- scikit-learn — LDA topic modeling, TF-IDF vectorization
- NLTK — stopwords, Brown Corpus comparison
