import sys
import nltk
from nltk.corpus import stopwords, brown
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import re

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('brown', quiet=True)
nltk.download('punkt', quiet=True)

# ── Load corpus ──────────────────────────────────────────────────────────────
try:
    with open("sample_corpus.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
except FileNotFoundError:
    print("ERROR: sample_corpus.txt not found. Please create it first.")
    sys.exit(1)

# Split into paragraphs (LDA needs multiple "documents")
paragraphs = [p.strip() for p in raw_text.split("\n\n") if len(p.strip()) > 50]
if len(paragraphs) < 2:
    paragraphs = [raw_text[i:i+500] for i in range(0, len(raw_text), 500)]

stop_words = list(stopwords.words("english"))

# ── Step 1: LDA ───────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("LDA TERM CANDIDATES (unfiltered)")
print("="*60)

cv = CountVectorizer(
    max_df=0.95, min_df=1,
    stop_words=stop_words,
    token_pattern=r'\b[a-zA-Z]{3,}\b'
)
dtm = cv.fit_transform(paragraphs)
vocab = cv.get_feature_names_out()

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(dtm)

lda_terms = set()
for topic in lda.components_:
    top_indices = topic.argsort()[-15:]
    for i in top_indices:
        lda_terms.add(vocab[i])

for term in sorted(lda_terms):
    print(f"  {term}")

# ── Step 2: TF-IDF filter ────────────────────────────────────────────────────
filter_mode = sys.argv[1] if len(sys.argv) > 1 else None

if filter_mode == "--filter" and len(sys.argv) > 2:
    mode = sys.argv[2]

    if mode == "tfidf":
        print("\n" + "="*60)
        print("TF-IDF FILTERED LIST")
        print("="*60)

        tfidf = TfidfVectorizer(
            max_df=0.95, min_df=1,
            stop_words=stop_words,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )
        tfidf_matrix = tfidf.fit_transform(paragraphs)
        tfidf_vocab = tfidf.get_feature_names_out()
        mean_scores = tfidf_matrix.mean(axis=0).A1

        # Keep terms that are both in LDA list AND have decent TF-IDF score
        threshold = sorted(mean_scores, reverse=True)[len(mean_scores)//8]
        tfidf_terms = {tfidf_vocab[i] for i, s in enumerate(mean_scores) if s >= threshold}
        filtered = lda_terms & tfidf_terms

        for term in sorted(filtered):
            print(f"  {term}")

    elif mode == "brown":
        print("\n" + "="*60)
        print("BROWN CORPUS COMPARISON LIST")
        print("="*60)

        brown_words = set(w.lower() for w in brown.words() if w.isalpha())
        # Terms in LDA list that are rare or absent in Brown Corpus
        brown_filtered = set()
        brown_freq = nltk.FreqDist(w.lower() for w in brown.words())
        threshold = 50  # words appearing fewer than 50x in Brown = domain-specific

        for term in lda_terms:
            if brown_freq[term] < threshold:
                brown_filtered.add(term)

        for term in sorted(brown_filtered):
            print(f"  {term}")