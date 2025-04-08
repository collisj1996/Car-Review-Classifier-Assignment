from typing import List

import pandas as pd
import nltk
import numpy as np
from nltk import downloader
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

downloader.download("stopwords")
downloader.download("punkt_tab")
english_stop_words = set(stopwords.words("english"))
all_stop_words = set()

stemmer = SnowballStemmer("english")

# add versions of a stopword with and without apostrophes
# e.g. "don't" and "dont"
for word in english_stop_words:
    if "'" in word:
        all_stop_words.add(word.replace("'", ""))
    else:
        all_stop_words.add(word)


def read_csv(file_path) -> pd.DataFrame:
    """
    Reads a CSV file and returns a DataFrame.
    """
    return pd.read_csv(file_path)


def preprocessor(text: str) -> List[str]:
    """Preprocesses text by removing punctuation and stopwords, stemming, and lowercasing and converts it to a list of tokens."""

    text = text.lower()
    tokens = nltk.word_tokenize(text)

    # remove punctuation
    tokens = [word for word in tokens if word.isalnum()]

    # remove stopwords
    tokens = [word for word in tokens if word not in all_stop_words]

    # stem words
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens


class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.array([preprocessor(text) for text in X])


def split_data(df: pd.DataFrame, absolute_test_size: int = 276) -> tuple:
    """
    Splits the DataFrame into training and testing sets.
    """
    train_df, test_df = train_test_split(
        df, test_size=absolute_test_size, random_state=120, stratify=df["Sentiment"]
    )
    return train_df, test_df


def lower_case(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a new DataFrame with all text in the 'Review' column converted to lowercase.
    """
    df_copy = df.copy()
    df_copy["Review"] = df_copy["Review"].str.lower()

    return df_copy


def redundant_token_removal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a new DataFrame with redundant tokens removed from the 'Review' column.
    """

    df_copy = df.copy()
    df_copy["Review"] = df_copy["Review"].apply(
        lambda x: " ".join([word for word in x.split() if word not in all_stop_words])
    )

    return df_copy


def remove_punctuation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a new DataFrame with punctuation removed from the 'Review' column.
    """
    df_copy = df.copy()
    df_copy["Review"] = df_copy["Review"].str.replace(r"[^\w\s]", "", regex=True)

    return df_copy


def stemming(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a new DataFrame with stemming applied to the 'Review' column.
    """
    df_copy = df.copy()
    df_copy["Review"] = df_copy["Review"].apply(
        lambda x: " ".join([stemmer.stem(word) for word in x.split()])
    )

    return df_copy


if __name__ == "__main__":
    # Read the CSV file
    input_data = read_csv("car-reviews.csv")

    test_size = 276  # 20% of 1382

    # Split the data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(
        input_data,
        test_size=test_size,
        random_state=120,
        stratify=input_data["Sentiment"],
    )

    pipeline = Pipeline(
        [
            ("preprocess", Preprocessor()),
            ("vectorize", CountVectorizer()),  # or TfidfVectorizer
            ("clf", MultinomialNB()),
        ]
    )
    # train_df_lower = lower_case(train_df)
    # test_df_lower = lower_case(test_df)

    # train_df_no_stop = redundant_token_removal(train_df_lower)
    # test_df_no_stop = redundant_token_removal(test_df_lower)

    # train_df_stemmed = stemming(train_df_no_stop)
    # test_df_stemmed = stemming(test_df_no_stop)
