import pandas as pd
import nltk
import torch
from torchtext.vocab import GloVe
from nltk import downloader
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

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

def preprocess_clean_text(text: str) -> str:

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
    tokens = [lemmatizer.lemmatize(word, pos_to_wordnet(pos)) for word, pos in token_tags]

    return tokens

def create_tensor_from_tokens(tokens, sentiment, glove, max_length=100):
    """Converts tokens to a tensor using GloVe embeddings."""

    # Convert tokens to indices using GloVe embeddings
    indices = [glove.stoi[token] for token in tokens if token in glove.stoi]
    label = 1 if sentiment == "Pos" else 0

    # Pad or truncate the sequence to the desired length
    if len(indices) < max_length:
        indices += [0] * (max_length - len(indices))  # Pad with zeros
    else:
        indices = indices[:max_length]  # Truncate

    return (torch.tensor(indices), torch.tensor(label).long())

def text_to_embeddings(input_data, glove):

    output_data = []

    for index, row in input_data.iterrows():
        tokens = preprocess_clean_text(row["Review"])
        sentiment = row["Sentiment"]
        tensor = create_tensor_from_tokens(tokens, sentiment, glove)
        output_data.append(tensor)

    return output_data


class RNNModel(torch.nn.Module):
    """PyTorch RNN model for sentiment analysis."""

    def __init__(self, input_size, hidden_size, layers_n, output_size, device, embedding):
        super(RNNModel, self).__init__()
        self.embedding = torch.nn.Embedding.from_pretrained(embedding.vectors)
        self.device = device # is this necessary?
        self.layers_n = layers_n
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, layers_n, batch_first=True)
        self.fully_connected = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x) # embedding layer

        # initial_hidden_state = torch.zeros(self.layers_n, x.size(0), self.hidden_size).to(self.device) # initial hidden state
        # initial_cell_state = torch.zeros(self.layers_n, x.size(0), self.hidden_size).to(self.device) # initial cell state

        out, _ = self.lstm(x) # LSTM output
        out = self.fully_connected(out)
        return out

def train_model(model, training_data, validation_data, epochs, learning_rate):
    
    loss_function = torch.nn.CrossEntropyLoss()
    optimization_function = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (review_embeddings, labels) in enumerate(training_data):

            optimization_function.zero_grad()

            # Forward pass
            outputs = model(review_embeddings)
            loss = loss_function(outputs[-1], labels)

            # Backward pass and optimization
            optimization_function.zero_grad()
            loss.backward()
            optimization_function.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

def run_rnn_model():

    input_data = pd.read_csv("car-reviews.csv")

    test_size = 276  # 20% of the data for testing

    training_data, test_data = train_test_split(
        input_data,
        test_size=test_size,
        random_state=120,
        stratify=input_data["Sentiment"], # stratify by sentiment, so that the test set has the same distribution of sentiments as the training set
    )

    # # preprocess the text data
    # training_data_processed = training_data["review"].apply(preprocessor)
    # test_data_processed = test_data["review"].apply(preprocessor)

    glove = GloVe(name="6B", dim=50)

    training_data_processed = text_to_embeddings(training_data, glove)
    test_data_processed = text_to_embeddings(test_data, glove)


    # convert the text data into a format suitable for RNNs (e.g., using embeddings)
    

    # define hyperparameters
    input_size = 50  # size of the input features (e.g., embedding size)   
    hidden_size = 128  # size of the hidden layer
    layers_n = 2  # number of layers in the RNN
    output_size = 2  # binary classification (positive/negative sentiment)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rnn_model = RNNModel(input_size, hidden_size, layers_n, output_size, device, glove)

    train_model(rnn_model, training_data_processed, test_data_processed, epochs=10, learning_rate=0.001)





if __name__ == "__main__":
    run_rnn_model()