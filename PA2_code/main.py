import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
from utilities import Utilities
# import nltk
# nltk.download('punkt')

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset
from transformer import TransformerClassifier, TransformerDecoderLM
from torch import nn

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 20 # 15 epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    # Below are modifications
    attention_masks = (padded_sequences != 0).long()  # Create attention masks
    return padded_sequences, labels, attention_masks


def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y, attention_masks in data_loader:
            # X, Y = X.to(device), Y.to(device)
            # outputs = classifier(X)
            X, Y, attention_masks = X.to(device), Y.to(device), attention_masks.to(device)
            outputs, _ = classifier(X, attention_masks) ##
            _, predicted = torch.max(outputs.data, 1) #
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0.0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss, _ = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    test_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset, batch_size=batch_size, collate_fn=collate_batch)

    # Initialize classifier
    model = TransformerClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_size=n_embd,
        num_heads=n_head,
        num_layers=n_layer,
        max_length=block_size,
        num_classes=n_output,
        hidden_dim=n_hidden,
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

     # for the classification  task, you will train for a fixed number of epochs like this:

    for epoch in range(epochs_CLS):
        total_loss = 0
        for xb, yb, attention_masks in train_CLS_loader:
            xb, yb, attention_masks= xb.to(device), yb.to(device), attention_masks.to(device)
            # CLS training code here
            optimizer.zero_grad()
            # Pass attention masks to the model
            outputs, _ = model(xb, attention_masks)  ##
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_CLS_loader)
        test_accuracy = compute_classifier_accuracy(model,test_CLS_loader)
        print(f"Epoch {epoch+1}/{epochs_CLS}, Loss: {avg_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Compute the number of parameters in the encoder
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    encoder_params = count_parameters(model.encoder)
    print(f"Number of parameters in the encoder: {encoder_params}")

    ## After training, perform sanity check using Utilities class
    utils = Utilities(tokenizer, model)
    test_sentence = "That is in Israel's interest, Palestine's interest, America's interest, and the world's interest."
    utils.sanity_check(test_sentence, block_size)

    print("-"*80)

    decoder_model = TransformerDecoderLM(
        vocab_size=tokenizer.vocab_size,
        embed_size=n_embd,
        num_heads=n_head,
        num_layers=n_layer,
        max_length=block_size,
        dropout=0.1
    ).to(device)

    decoder_optimizer = torch.optim.Adam(decoder_model.parameters(), lr=learning_rate)

    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    # Prepare test datasets
    test_files = ['test_LM_hbush.txt', 'test_LM_wbush.txt', 'test_LM_obama.txt']
    test_LM_datasets = {}
    for test_file in test_files:
        test_inputfile = f"speechesdataset/{test_file}"
        with open(test_inputfile, 'r', encoding='utf-8') as f:
            test_text = f.read()
        test_dataset = LanguageModelingDataset(tokenizer, test_text, block_size)
        test_LM_datasets[test_file] = DataLoader(test_dataset, batch_size, shuffle=False)

    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)
        #LM training code here
        decoder_model.zero_grad()
        loss, _ = decoder_model(xb, yb)
        loss.backward()
        decoder_optimizer.step()

        # Evaluate and report every eval_interval iterations
        if (i + 1) % eval_interval == 0 or i == 0:
            train_perplexity = torch.exp(loss).item()
            print(f"Iteration {i+1}/{max_iters}, Loss: {loss.item():.4f}, Train Perplexity: {train_perplexity:.2f}")

            # Evaluate on test sets
            for test_file, test_loader in test_LM_datasets.items():
                test_perplexity = compute_perplexity(decoder_model, test_loader, eval_iters=eval_iters)
                print(f"Perplexity on {test_file}: {test_perplexity:.2f}")
            print('-' * 80)

    # After training, perform sanity check using Utilities class
    utils_dec = Utilities(tokenizer, decoder_model)
    test_sentence_dec = "Through this remarkable chapter in the history of the United States and Iraq, we have met our responsibility."
    utils_dec.sanity_check(test_sentence_dec, block_size)

    # Compute the number of parameters in the decoder
    decoder_params = count_parameters(decoder_model)
    print(f"Number of parameters in the decoder: {decoder_params}")



if __name__ == "__main__":
    main()
