import pandas as pd
import nltk
from nltk import downloader
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, confusion_matrix

downloader.download("stopwords")
downloader.download("punkt_tab")
downloader.download("wordnet")
downloader.download('averaged_perceptron_tagger_eng')

english_stop_words = set(stopwords.words("english"))
all_stop_words = set()

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

# add versions of a stopword with and without apostrophes
# e.g. "don't" and "dont"
for word in english_stop_words:
    if "'" in word:
        all_stop_words.add(word.replace("'", ""))
    else:
        all_stop_words.add(word)

input_data = pd.read_csv("car-reviews.csv")

def preprocessor(text: str) -> str:
    """Preprocesses text by removing punctuation and stopwords, stemming, and lowercasing."""

    text = preprocess_remove_punctuation(text)
    text = preprocess_lowercase(text)
    text = preprocess_remove_stopwords(text)
    text = preprocess_stem(text)
    return text


def preprocess_lowercase(text: str) -> str:
    """Preprocesses text by lowercasing."""

    text = text.lower()
    return text

def preprocess_remove_punctuation(text: str) -> str:
    """Preprocesses text by removing punctuation """

    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    return " ".join(tokens)

def preprocess_remove_stopwords(text: str) -> str:
    """Preprocesses text by removing stopwords."""

    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in all_stop_words]
    return " ".join(tokens)

def preprocess_stem(text: str) -> str:
    """Preprocesses text by stemming."""

    def pos_to_wordnet(pos: str) -> str:
        """Converts POS tag to WordNet format."""

        first_char = pos[0]

        if first_char == "J":
            return "a"
        elif first_char == "V":
            return "v"
        elif first_char == "N":
            return "n"
        elif first_char == "R":
            return "r"
        else:
            return "n"

    tokens = nltk.word_tokenize(text)
    token_tags = pos_tag(tokens)

    # tokens = [stemmer.stem(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word, pos_to_wordnet(pos)) for word, pos in token_tags]
    return " ".join(tokens)

class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [preprocessor(text) for text in X]


def run_ml_pipeline(input_data: pd.DataFrame) -> None:
    """
    Runs the machine learning pipeline on input dataframe.
    """
    test_size = 276  # 20% of the data for testing

    training_data, test_data = train_test_split(
        input_data,
        test_size=test_size,
        random_state=120,
        stratify=input_data["Sentiment"],
    )

    # ----- A note about unseen words in the test set ------:
    # The MultinomialNB Classifier is passed an alpha value of 1.0, which happens to be the default value.
    # This is the additive laplace smoothing parameter and means that all features will have a constant value of 1.0 added to them.
    # This will essentially mean when an unseen word is encountered in the test set, it will be treated as if it has been seen once in the training set.
    # This is much better than the unseen word having a probability of 0.0 which would cause Naive Bayes to be unable to classify the review.
    pipeline = Pipeline(
        [
            ("preprocess", Preprocessor()),
            ("vectorize", CountVectorizer()),
            ("classify", MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)),
        ]
    )

    pipeline.fit(training_data["Review"], training_data["Sentiment"])

    predictions = pipeline.predict(test_data["Review"])

    accuracy = accuracy_score(test_data["Sentiment"], predictions)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    labels = ["Pos", "Neg"]

    cm = confusion_matrix(test_data["Sentiment"], predictions, labels=labels)

    cm_normalized = cm.astype("float") / cm.sum()

    print("Confusion Matrix Proportions:")
    print(f"True Negative: {cm_normalized[0][0] * 100:.2f}%")
    print(f"False Negative: {cm_normalized[1][0] * 100:.2f}%")
    print(f"True Positive: {cm_normalized[1][1] * 100:.2f}%")
    print(f"False Positive: {cm_normalized[0][1] * 100:.2f}%")


def words_and_punctuation_removal_example():
    """
    Example of stopword removal, punctuation removal, and lowercasing, as part of the marking criteria.
    """

    # 2 example rows from car-reviews.csv
    example_reviews = input_data["Review"].head(2).tolist()

    # additional made-up test review with punctuation
    example_text = "This is a sample review! It's great, isn't it?"
    example_reviews.append(example_text)

    for text in example_reviews:
        processed_review = preprocess_remove_punctuation(text)
        processed_review = preprocess_lowercase(processed_review)
        processed_review = preprocess_remove_stopwords(processed_review)

        print(f"Original: \n {text} \n")
        print(f"Processed: \n {processed_review} \n")
        print("----------------------\n")


def preprocessor_example():
    """Example of preprocessor function usage."""

    examples = []

    # additional made-up test review with punctuation
    stem_example_1 = "Here's some words with the same root stems! Running, run, runs."
    stem_example_2 = "... And some more! Removed, remove, removing."
    stem_example_3 = "... And more! Driving, Drove, Drive."

    examples.append(stem_example_1)
    examples.append(stem_example_2)
    examples.append(stem_example_3)

    for text in examples:
        processed_review = preprocessor(text)
        print(f"Original: \n {text} \n")
        print(f"Processed: \n {processed_review} \n")
        print("----------------------\n")


def bag_of_words_vector_example():
    """
    Example of bag-of-words based vectorization using the CountVectorizer from sklearn.
    """

    class DebugStep(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            print("DebugStep: Transforming data...")
            self.previous_data = X
            return X

    test_size = 276  # 20% of the data for testing

    training_data, test_data = train_test_split(
        input_data,
        test_size=test_size,
        random_state=120,
        stratify=input_data["Sentiment"],
    )

    pipeline = Pipeline(
        [
            ("preprocess", Preprocessor()),
            ("preprocess_debug", DebugStep()),
            ("vectorize", CountVectorizer()),
            ("vectorize_debug", DebugStep()),
            ("classify", MultinomialNB()),
        ]
    )

    pipeline.fit(training_data["Review"], training_data["Sentiment"])

    # analyse preprocessor
    preprocessor = pipeline.named_steps["preprocess"]
    preprocessor_output = pipeline.named_steps["preprocess_debug"].previous_data

    # anaylse CountVectorizer
    vectorizer = pipeline.named_steps["vectorize"]
    vocabulary = vectorizer.vocabulary_
    index_to_word = {index: word for word, index in vocabulary.items()}

    # get output of CountVectorizer
    vectorizer_output = pipeline.named_steps["vectorize_debug"].previous_data

    print("Vectorizer Output Shape:")
    print(vectorizer_output.shape)

    print("Vocabulary Length:")
    print(len(vocabulary))

    # first 5 review vectors
    for i in range(5):
        indices = vectorizer_output[i].indices
        vector_for_row = vectorizer_output.A[i]

        word_to_count = {word: 0 for word in index_to_word.values()}

        for indicy in indices:
            word_to_count[index_to_word[indicy]] = vector_for_row[indicy]

        # add to data frame for nice printing
        vector_to_vocabulary = pd.DataFrame(word_to_count, index=[0])

        print(f"###### Review {i} vector ######\n")
        print("Processed Text:\n")
        print(preprocessor_output[i])
        print("\nVector:\n")
        print(vector_to_vocabulary)
        print("\n----------------------\n")

if __name__ == "__main__":
    # words_and_punctuation_removal_example()
    # preprocessor_example()
    run_ml_pipeline(input_data)
    # bag_of_words_vector_example()
