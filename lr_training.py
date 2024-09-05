import torch
import joblib
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class ProteinDataset(Dataset):
    def __init__(self, sequences, labels, tokenizer, max_length):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = ' '.join(self.sequences[idx])  # Add space between amino acids
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            sequence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Load your data
df = pd.read_csv('dataset_terpenos.csv')

# Split into sequences and labels
sequences = df['sequence'].values
labels = df['label'].values

# Tokenizer and model initialization
tokenizer = BertTokenizer.from_pretrained('rostlab/prot_bert_bfd')

# Create dataset
train_dataset = ProteinDataset(sequences, labels, tokenizer, max_length=512)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Loading ProtBert model and tokenizer...')
tokenizer = BertTokenizer.from_pretrained('rostlab/prot_bert_bfd', do_lower_case=False)
model = BertModel.from_pretrained('./terpene_model')
model.to(device)


def extract_embeddings(model, tokenizer, sequences, max_length=512):
    embeddings = []
    model.eval()
    # with torch.no_grad():
    for sequence in sequences:
        sequence = ' '.join(sequence)
        sequence = sequence.replace('U', 'X')
        sequence = sequence.replace('O', 'X')
        sequence = sequence.replace('B', 'X')
        sequence = sequence.replace('Z', 'X')

        tokenized_sequence = tokenizer.encode_plus(sequence, add_special_tokens=True, max_length=20000, truncation=True)
        input_ids = torch.tensor([tokenized_sequence['input_ids']]).to(device)
        attention_mask = torch.tensor([tokenized_sequence['attention_mask']]).to(device)

        with torch.no_grad():
            last_hidden_states = model(input_ids, attention_mask=attention_mask)[0]

        embedding = last_hidden_states[0].cpu().numpy()
        seq_len = (attention_mask[0] == 1).sum()
        seq_embedding = embedding[1:seq_len-1]
        mean_pool = np.mean(seq_embedding, axis=0)

        embeddings.append(mean_pool)

    return np.array(embeddings)

# Extract embeddings
embeddings = extract_embeddings(model, tokenizer, sequences)
print(embeddings.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)

print(X_train.shape)
print(X_test.shape)

# Train logistic regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Evaluate
y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Save the model
joblib.dump(lr, 'terpene_lr_model.pkl')
