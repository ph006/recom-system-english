# Resume Recommendation System (English) — NLP

A text-search / recommendation system that takes a short free-text query
(e.g. *"a data scientist with a lot of experience"*) and returns the most relevant
English resume-style texts from a dataset.

It combines two classic NLP techniques:

1. **Word2Vec** — expands the query with semantically related words.
2. **TF-IDF + cosine similarity** — ranks every document against the expanded query.

> Everything lives in one notebook: [`Recom_english.ipynb`](Recom_english.ipynb).
> A Persian version of this project is in a separate repo (same pipeline, Persian text handling).

---

## How it works

```
query ──► clean ──► stem ──► Word2Vec: add similar words ──► TF-IDF vector ─┐
                                                                            ├─► cosine similarity ─► top-N documents
dataset text ──► clean ──► stem ───────────────────────────► TF-IDF matrix ─┘
```

### 1. Data loading
- `pii_dataset.csv` — main dataset, one document per row in a `text` column
  (people describing their profession and experience).
- `eng_stp_words` — CSV with a `words` column containing English stop words (127 words, NLTK-style).
- Missing text is replaced with `''`; all columns are cast to `str`.

### 2. Stop-word list = NLTK words + most frequent corpus words
`CountVectorizer` counts every word in the corpus. The **top 100** most frequent words are appended
to the stop-word list, on top of the standard ones. These are words so common in this dataset
("my", "experience"-style filler) that they carry no search value.

The author notes that going much further (e.g. 500 words) hurt the Word2Vec model, because
useful words were removed too.

### 3. Cleaning — `remove_word_eng(text)`
- Collapse whitespace (`\s+` → single space).
- Remove special characters (`[^\w\s]`).
- Drop any word (case-insensitive) that is in the stop-word list.

### 4. Stemming — `apply_nltk_stemmer(text)`
NLTK's `PorterStemmer` reduces words to their root (*analyzing → analyz*, *statistics → statist*),
so different forms of the same word match each other.

### 5. Query expansion — `word2vec(column, cleaner, user_input, num_results=10)`
1. Clean + stem the whole text column and tokenize it.
2. Train `Word2Vec` (`vector_size=150`, `window=6`, `min_count=10`, `workers=4`).
3. Clean + stem the user query.
4. For every query word longer than 3 characters, take its **5 most similar words**.
   If a similar word appears for several query words, its scores are **summed**.
5. Keep only similar words with a score **≥ 0.7**.
6. Return `"<query> <similar words…>"` as one string.

Example for `"a data scientist with alot of experience"` →
`dataset, predict, statist, model, analyz, institut, biologist, intellig, field, comput …`

### 6. Ranking — `tf_idf(X_column, user_input, num_results=10)`
1. Build the TF-IDF matrix of the cleaned + stemmed corpus.
2. Build the expanded query (original words + Word2Vec words, stemmed, words ≤ 3 chars dropped).
3. Transform the query into the same TF-IDF space.
4. Compute **cosine similarity** against every document and return the top `num_results` rows.

Because the original query words appear twice in the expanded query, they get extra weight, so
the similar words support the search instead of overpowering it.

---

## Run it

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
jupyter notebook Recom_english.ipynb
```

Put `pii_dataset.csv` and `eng_stp_words` in the same folder as the notebook (they are not
included in this repo — see `.gitignore`).

Then run:

```python
tf_idf(df['text'], 'a data scientist with alot of experience')
```

## Notes & known limitations
- Word2Vec is **retrained on every call** (slow on big data). Train once and reuse the model for real use.
- `tf_idf` returns rows from the global `df`, not from the `X_column` argument — keep them the same.
- `word2vec` prints the raw similarity dictionary and `tf_idf` prints the final query (debug output).
- The notebook comment says "top 30 words" but the code takes the top **100**.
- A bigger/denser dataset gives noticeably higher similarity scores than the small Persian one.
- The dataset contains personal-style details; check its license/privacy before publishing it.
