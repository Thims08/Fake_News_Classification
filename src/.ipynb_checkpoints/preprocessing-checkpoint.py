import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocessing(df):
    df_cleaned = df.dropna(subset=['text','title'])
    df_cleaned = df_cleaned.drop_duplicates(subset=['text'])
    df_cleaned = df_cleaned.drop_duplicates(subset=['title'])

    df_cleaned['length_title'] = df_cleaned['title'].str.len()
    df_cleaned = df_cleaned[df_cleaned['length_title'] > 1]

    df_cleaned['length_text'] = df_cleaned['text'].str.len()
    df_cleaned = df_cleaned[df_cleaned['length_text'] > 10]

    return df_cleaned


def text_preprocessing(df):
    df['text'] = df['title'] + " " + df['text']

    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def clean_text(text):
        text = re.sub(r'[^a-zA-Z]', ' ', str(text))
        text = text.lower()
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word)
                  for word in tokens if word not in stop_words]
        return ' '.join(tokens)

    df['cleaned_text'] = df['text'].apply(clean_text)
    return df