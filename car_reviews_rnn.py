from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import nltk
import torch
from torchtext.vocab import GloVe
from nltk import downloader
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split

downloader.download("stopwords")
downloader.download("punkt_tab")
downloader.download("wordnet")

english_stop_words = set(stopwords.words("english"))
all_stop_words = set()

lemmatizer = WordNetLemmatizer()

# add versions of a stopword with and without apostrophes
# e.g. "don't" and "dont"
for word in english_stop_words:
    if "'" in word:
        all_stop_words.add(word.replace("'", ""))
    else:
        all_stop_words.add(word)


class RNNModel(torch.nn.Module):
    """PyTorch RNN model for sentiment analysis."""

    def __init__(
        self, input_size, hidden_size, layers_n, output_size, device, embedding
    ):
        super(RNNModel, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(embedding.vectors)
        self.device = device  # is this necessary?
        self.layers_n = layers_n
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, layers_n, batch_first=True)
        self.fully_connected = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)  # embedding layer

        # initial_hidden_state = torch.zeros(self.layers_n, x.size(0), self.hidden_size).to(self.device) # initial hidden state
        # initial_cell_state = torch.zeros(self.layers_n, x.size(0), self.hidden_size).to(self.device) # initial cell state

        out, _ = self.lstm(x)  # LSTM output
        last_hidden = out[:, -1, :]
        fc_out = self.fully_connected(last_hidden)

        # out has shape (batch_size, seq_len, output_size)
        # we want the last output for each sequence, so we take the last time step
        # out = out[:, -1, :]
        return fc_out


@dataclass
class Hyperparameters:
    input_size: int = 50
    hidden_size: int = 50
    layers_n: int = 1
    output_size: int = 2
    epochs: int = 100
    learning_rate: float = 0.001
    max_length: int = 150


class CarReviewClassifier:
    def __init__(
        self,
        training_hyperparameters: Hyperparameters = Hyperparameters(),
        all_data_path="car-reviews.csv",
    ):
        self.glove = GloVe(
            name="6B", dim=50
        )  # pretrained GloVe embeddings, note this is a 800MB download (it will be cached)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyperparameters = training_hyperparameters
        self.model = None

        self.all_data = pd.read_csv(all_data_path)

        training_data_unprocessed, validation_data_unprocessed = self._get_train_test_split()

        training_vectors = self._pipeline_preprocess_and_vectorise(
            training_data_unprocessed
        )
        validation_vectors = self._pipeline_preprocess_and_vectorise(
            validation_data_unprocessed
        )

        self.training_data = torch.utils.data.DataLoader(
            training_vectors, batch_size=len(training_vectors), shuffle=True
        )
        self.validation_data = torch.utils.data.DataLoader(
            validation_vectors, batch_size=len(validation_vectors), shuffle=True
        )

    def _get_train_test_split(self):
        test_size = 276  # 20% of the data for testing, using absolute number of samples

        # Split the data into training and validation sets
        training_data_unprocessed, validation_data_unprocessed = (
            train_test_split(
                self.all_data,
                test_size=test_size,
                random_state=120,
                stratify=self.all_data[
                    "Sentiment"
                ],  # stratify by sentiment, so that the test set has the same distribution of sentiments as the training set
            )
        )

        return training_data_unprocessed, validation_data_unprocessed

    def _preprocess_clean_text(self, text: str) -> str:
        """Preprocesses and cleans the input text."""

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

        text = text.lower()

        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum()]
        tokens = [word for word in tokens if word not in all_stop_words]
        token_tags = pos_tag(tokens)
        tokens = [
            lemmatizer.lemmatize(word, pos_to_wordnet(pos)) for word, pos in token_tags
        ]

        return tokens

    def _create_tensor_from_tokens(self, tokens, sentiment):
        """Converts tokens to a tensor using GloVe embeddings."""

        # Convert tokens to indices using GloVe embeddings
        indices = [
            self.glove.stoi[token] for token in tokens if token in self.glove.stoi
        ]
        label = 1 if sentiment == "Pos" else 0

        # Pad or truncate the sequence to the desired length
        if len(indices) < self.hyperparameters.max_length:
            indices += [0] * (
                self.hyperparameters.max_length - len(indices)
            )  # Pad with zeros
        else:
            indices = indices[: self.hyperparameters.max_length]  # Truncate

        return (torch.tensor(indices), torch.tensor(label).long())

    def _pipeline_preprocess_and_vectorise(self, input_data):
        """Pipeline that pre-procesesses and vectorizes the input data."""
        output_data = []

        print("Preprocessing and vectorizing data...")

        for index, row in input_data.iterrows():
            tokens = self._preprocess_clean_text(row["Review"])
            tensor = self._create_tensor_from_tokens(tokens, row["Sentiment"])
            output_data.append(tensor)

        return output_data

    def _init_new_model(self):
        self.model = RNNModel(
            self.hyperparameters.input_size,
            self.hyperparameters.hidden_size,
            self.hyperparameters.layers_n,
            self.hyperparameters.output_size,
            self.device,
            self.glove,
        )

    def set_training_hyperparameters(self, hyperparameters: Hyperparameters):
        self.hyperparameters = hyperparameters

    def load_model(self, model_name):
        self.model = torch.load(f"{model_name}.pth")

    def save_model(self, model_name):
        torch.save(self.model, f"{model_name}.pth")

    def train(self):
        self._init_new_model()

        loss_function = torch.nn.CrossEntropyLoss()
        optimization_function = torch.optim.Adam(
            self.model.parameters(), lr=self.hyperparameters.learning_rate
        )

        # training_reviews, training_labels = map(list, zip(*self.training_data))

        data_loader = torch.utils.data.DataLoader(
            dataset=self.training_data, batch_size=len(self.training_data), shuffle=True
        )

        validation_loader = torch.utils.data.DataLoader(
            dataset=self.validation_data, batch_size=len(self.validation_data), shuffle=True
        )

        for epoch in range(self.hyperparameters.epochs):
            # for i, (review_embeddings, labels) in enumerate(self.training_data):
            for reviews, labels in data_loader:

                optimization_function.zero_grad()
                # Forward pass
                outputs = self.model(reviews)
                loss = loss_function(outputs, labels)

                # Backward pass and optimization
                loss.backward()
                optimization_function.step()

            training_accuracy = self.accuracy(data_loader)
            validation_accuracy = self.accuracy(validation_loader)

            print(
                f"Epoch [{epoch + 1}/{self.hyperparameters.epochs}], "
                f"Training Loss: {loss.item():.4f}, "
                f"Training Accuracy: {training_accuracy:.4f}, "
                f"Validation Accuracy: {validation_accuracy:.4f}"
            )

    def accuracy(self, data_loader):
        correct, total = 0, 0
        for reviews, sentiments in data_loader:
            output = self.model(reviews)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(sentiments.view_as(pred)).sum().item()
            total += sentiments.shape[0]
        return correct / total


if __name__ == "__main__":

    
    # mode 1 
    crc = CarReviewClassifier()
    crc.train()
    crc.save_model("test_2")
    
    # mode 2
    # crc = CarReviewClassifier()
    # crc.load_model("test")
    # crc.accuracy()
