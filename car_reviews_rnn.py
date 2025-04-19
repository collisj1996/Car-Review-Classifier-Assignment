from dataclasses import dataclass

import pandas as pd
import nltk
import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import matplotlib.pyplot as plt
from torchtext.vocab import GloVe
from nltk import downloader
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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
    """PyTorch based RNN model"""

    def __init__(
        self,
        input_size,
        hidden_size,
        layers_n,
        output_size,
        device,
        embedding,
    ):
        super(RNNModel, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(embedding.vectors)
        self.device = device
        self.layers_n = layers_n
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, layers_n, batch_first=True)
        self.fully_connected = torch.nn.Linear(hidden_size, output_size)

    def forward(self, input_data, text_lengths):
        input_data = input_data.to(self.device)
        # text_lengths = torch.tensor(text_lengths).to(self.device)

        emdedded_data = self.embedding(input_data)  # embedding layer

        initial_hidden_state = torch.zeros(
            self.layers_n, emdedded_data.size(0), self.hidden_size
        ).to(
            self.device
        )  # initial hidden state
        initial_cell_state = torch.zeros(self.layers_n, emdedded_data.size(0), self.hidden_size).to(
            self.device
        )  # initial cell state

        packed_embedded_data = torch.nn.utils.rnn.pack_padded_sequence(
            emdedded_data, text_lengths, batch_first=True, enforce_sorted=False
        )

        rnn_out, (hidden_state, cell_state) = self.lstm(packed_embedded_data, (initial_hidden_state, initial_cell_state))  # LSTM output

        last_hidden = hidden_state[-1]
        fc_out = self.fully_connected(last_hidden)
        return fc_out.squeeze(1) 


@dataclass
class Hyperparameters:
    input_size: int = 50
    hidden_size: int = 50
    layers_n: int = 1
    epochs: int = 200
    learning_rate: float = 0.001
    max_length: int = 500
    early_stop_retry_threshold: int = 10


class CarReviewClassifier:
    def __init__(
        self,
        training_hyperparameters: Hyperparameters = Hyperparameters(),
        all_data_path="car-reviews.csv",
        torch_seed=None,
        use_debug_cache=False,
    ):
        self.glove = GloVe(
            name="6B", dim=50
        )  # pretrained GloVe embeddings, note this is a 800MB download (it will be cached)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyperparameters = training_hyperparameters
        self.model = None

        if torch_seed is not None:
            torch.manual_seed(torch_seed)

        self.all_data = pd.read_csv(all_data_path)

        if use_debug_cache:
            # check if the preprocessed data files exist
            try:
                # load the preprocessed data from the debug_cache folder
                self.training_x = torch.load("debug_cache/training_x.pt")
                self.validation_x = torch.load("debug_cache/validation_x.pt")
                self.training_y = torch.load("debug_cache/training_y.pt")
                self.validation_y = torch.load("debug_cache/validation_y.pt")
                self.train_lengths = torch.load("debug_cache/train_lengths.pt")
                self.validation_lengths = torch.load("debug_cache/validation_lengths.pt")

                if self.device.type == "cuda":
                    self.training_x = self.training_x.cuda()
                    self.validation_x = self.validation_x.cuda()
                    self.training_y = self.training_y.cuda()
                    self.validation_y = self.validation_y.cuda()
                return
            except Exception:
                pass
            
        training_data_unprocessed, validation_data_unprocessed = (
            self._get_train_test_split()
        )

        self.training_y = self._get_label_tensors(
            training_data_unprocessed["Sentiment"]
        )
        self.validation_y = self._get_label_tensors(
            validation_data_unprocessed["Sentiment"]
        )

        self.training_x, self.train_lengths = self._pipeline_preprocess_and_vectorise(
            training_data_unprocessed
        )
        self.validation_x, self.validation_lengths = self._pipeline_preprocess_and_vectorise(
            validation_data_unprocessed
        )

        if self.device.type == "cuda":
            self.training_x = self.training_x.cuda()
            self.validation_x = self.validation_x.cuda()
            self.training_y = self.training_y.cuda()
            self.validation_y = self.validation_y.cuda()

        if use_debug_cache:
            # save the preprocessed data to a file for later use
            torch.save(self.training_x, "debug_cache/training_x.pt")
            torch.save(self.validation_x, "debug_cache/validation_x.pt")
            torch.save(self.training_y, "debug_cache/training_y.pt")
            torch.save(self.validation_y, "debug_cache/validation_y.pt")
            torch.save(self.train_lengths, "debug_cache/train_lengths.pt")
            torch.save(self.validation_lengths, "debug_cache/validation_lengths.pt")

    def _get_label_tensors(self, labels):
        """Converts labels to tensors."""
        label_tensors = []

        for label in labels:
            if label == "Pos":
                label_tensors.append(torch.tensor(1.0))
            else:
                label_tensors.append(torch.tensor(0.0))

        return torch.stack(label_tensors)

    def _get_train_test_split(self):
        test_size = 276  # 20% of the data for testing, using absolute number of samples

        # Split the data into training and validation sets
        training_data_unprocessed, validation_data_unprocessed = train_test_split(
            self.all_data,
            test_size=test_size,
            random_state=120,
            stratify=self.all_data[
                "Sentiment"
            ],  # stratify by sentiment, so that the test set has the same distribution of sentiments as the training set
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

    def _create_tensor_from_tokens(self, tokens):
        """Converts tokens to a tensor using GloVe embeddings."""

        # Convert tokens to indices using GloVe embeddings
        indices = [
            self.glove.stoi[token] for token in tokens if token in self.glove.stoi
        ]

        return torch.tensor(indices)

    def _pipeline_preprocess_and_vectorise(self, input_data):
        """Pipeline that pre-procesesses and vectorizes the input data."""
        tensors = []
        lengths = []

        print("Preprocessing and vectorizing data...")

        for index, row in input_data.iterrows():
            tokens = self._preprocess_clean_text(row["Review"])
            tensor = self._create_tensor_from_tokens(tokens)

            # if len(tensor) > self.hyperparameters.max_length:
            #     tensor = tensor[: self.hyperparameters.max_length]

            tensors.append(tensor)
            lengths.append(len(tensor))

        padded_tensors = torch.nn.utils.rnn.pad_sequence(
            tensors, batch_first=True
        )

        return padded_tensors, lengths

    def _init_new_model(self):
        self.model = RNNModel(
            input_size=self.hyperparameters.input_size,
            hidden_size=self.hyperparameters.hidden_size,
            layers_n=self.hyperparameters.layers_n,
            output_size=1, # binary classification
            device=self.device,
            embedding=self.glove,
        )

    def set_training_hyperparameters(self, hyperparameters: Hyperparameters):
        self.hyperparameters = hyperparameters

    def load_model(self, model_name):
        self.model = torch.load(f"{model_name}.pth")

    def save_model(self, model_name):
        torch.save(self.model, f"{model_name}.pth")

    def train(self):
        self._init_new_model()
        self.model.to(self.device)

        loss_function = torch.nn.BCEWithLogitsLoss()
        optimization_function = torch.optim.Adam(
            self.model.parameters(), lr=self.hyperparameters.learning_rate
        )

        best_model = None
        best_val_cost = float("inf")
        early_stop_retry_count = 0

        for epoch in range(self.hyperparameters.epochs):
            # for train_reviews, train_sentiments in self.training_data:

            optimization_function.zero_grad()
            # Forward pass
            outputs = self.model(self.training_x, self.train_lengths)
            loss = loss_function(outputs, self.training_y)

            # Backward pass and optimization
            loss.backward()
            optimization_function.step()

            # for validation_reviews, validation_sentiments in self.validation_data:
            val_loss = loss_function(
                self.model(self.validation_x, self.validation_lengths), self.validation_y
            )

            # Use early stopping to prevent overfitting
            if val_loss < best_val_cost:
                early_stop_retry_count = 0
                best_val_cost = val_loss
                best_model = self.model.state_dict()
                print(f"Best model saved at epoch {epoch + 1}")
            else:
                early_stop_retry_count += 1
                if early_stop_retry_count >= self.hyperparameters.early_stop_retry_threshold:
                    print("Early stopping triggered.")
                    break

            training_accuracy = self.accuracy(self.training_x, self.training_y, self.train_lengths)
            validation_accuracy = self.accuracy(self.validation_x, self.validation_y, self.validation_lengths)

            print(
                f"Epoch [{epoch + 1}/{self.hyperparameters.epochs}], "
                f"Training Loss: {loss.item():.4f}, "
                f"Training Accuracy: {training_accuracy:.4f}, "
                f"Validation Accuracy: {validation_accuracy:.4f}"
            )

        # Save the best model
        if best_model is not None:
            self.model.load_state_dict(best_model)
            print("Best model loaded.")

    def accuracy(self, data, labels, packed_lengths):
        """Calculates the accuracy of the model on the given data."""

        total = len(data)
        output = self.model(data, packed_lengths)
        correct = (output > 0.5).eq(labels.view_as(output)).sum().item()
            
        print(f"Accuracy: {correct}/{total} = {correct / total:.4f}")
        return correct / total
    
    def confusion_matrix(self, data, labels, packed_lengths):
        """Calculates and displays the confusion matrix."""

        display_labels = ["Pos", "Neg"]

        output = self.model(data, packed_lengths)
        predictions = (output > 0.5).int().view_as(labels)

        cm = confusion_matrix(labels.cpu(), predictions.cpu())
        cmd = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=display_labels
        )
        cmd.plot()
        plt.show()


class HyperparameterSearch:
    def __init__(self):
        self.crc = CarReviewClassifier()
        self.best_hyperparameters = None
        self.best_validation_accuracy = 0.0
        self.history = []

    def search(self):
        # Implement hyperparameter search logic here
        search_space = {
            "hidden_size": [50, 100, 150],
            "layers_n": [1],
            "epochs": [50],
            "learning_rate": [0.001],
        }

        self._dfsearch(search_space, 0)

    def _dfsearch(self, search_space, depth, current_hyperparameters=None):
        if current_hyperparameters is None:
            current_hyperparameters = {}
            for key in search_space.keys():
                current_hyperparameters = {
                    key: value[0] for key, value in search_space.items()
                }
    
        if depth == len(search_space):
            hyperparameters = Hyperparameters(
                **current_hyperparameters
            )
            self.crc.set_training_hyperparameters(hyperparameters)
            self.crc.train()
            validation_accuracy = self.crc.accuracy(self.crc.validation_x, self.crc.validation_y, self.crc.validation_lengths)

            history_entry = {
                "hyperparameters": current_hyperparameters.copy(),
                "validation_accuracy": validation_accuracy,
            }

            self.history.append(history_entry)

            if validation_accuracy > self.best_validation_accuracy:
                self.best_hyperparameters = current_hyperparameters.copy()
                self.best_validation_accuracy = validation_accuracy

            return

        for key, values in search_space.items():
            for value in values:
                current_hyperparameters[key] = value
                self._dfsearch(search_space, depth + 1, current_hyperparameters)



def run_final_model():
    crc = CarReviewClassifier()
    crc.load_model("final")
    crc.accuracy(crc.validation_x, crc.validation_y, crc.validation_lengths)


if __name__ == "__main__":

    # mode 1
    crc = CarReviewClassifier(use_debug_cache=True, torch_seed=1)
    crc.train()
    crc.save_model("test_2")
    crc.confusion_matrix(crc.validation_x, crc.validation_y, crc.validation_lengths)

    # mode 2
    # crc = CarReviewClassifier()
    # crc.load_model("test")
    # crc.accuracy(crc.validation_x, crc.validation_y, crc.validation_lengths)
    # crc.confusion_matrix(crc.validation_x, crc.validation_y, crc.validation_lengths)

    # hyperparameter search
    # hps = HyperparameterSearch()
    # hps.search()
    # print("Best hyperparameters:", hps.best_hyperparameters)
    # print("Best validation accuracy:", hps.best_validation_accuracy)
