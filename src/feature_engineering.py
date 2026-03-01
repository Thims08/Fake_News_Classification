from sklearn.feature_extraction.text import TfidfVectorizer

def embedding(df, method="tfidf"):
    if method == 'tfidf':
        vectorizer = TfidfVectorizer(
            max_df=0.8,
            min_df=5,
            stop_words='english'
        )
        X = vectorizer.fit_transform(df['cleaned_text'])
        return X
    else:
        raise ValueError(f"Unknown embedding method: {method}")