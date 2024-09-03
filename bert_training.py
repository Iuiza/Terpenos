from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

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

model = BertForSequenceClassification.from_pretrained('rostlab/prot_bert_bfd', num_labels=2)
model.to(device)

# Adjusted training arguments to avoid errors
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="no",  # No evaluation during training
    save_strategy="steps",     # Save checkpoints every few steps
    save_steps=1000,
    save_total_limit=2,
    load_best_model_at_end=False  # Disable loading the best model at the end
)

from transformers import Trainer

class ContiguousTrainer(Trainer):
    def _save(self, output_dir: str, state_dict=None):
        # Make sure all tensors in the model are contiguous before saving
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()

        super()._save(output_dir, state_dict)

# Use the custom trainer to handle non-contiguous tensors
trainer = ContiguousTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./terpene_model')
tokenizer.save_pretrained('./terpene_model')

# def extract_embeddings(model, tokenizer, sequences, max_length=512):
#     embeddings = []
#     model.eval()
#     # with torch.no_grad():
#     for sequence in sequences:
#         # sequence = ' '.join(sequence)
#         # tokenized_sequence = tokenizer.encode_plus(sequence, add_special_tokens=True, max_length=max_length, truncation=True)
#         # input_ids = torch.tensor([tokenized_sequence['input_ids']]).to(device)
#         # attention_mask = torch.tensor([tokenized_sequence['attention_mask']]).to(device)

#         # last_hidden_states = model(input_ids, attention_mask=attention_mask)[0]
#         # seq_len = (attention_mask[0] == 1).sum()
#         # seq_embedding = last_hidden_states[0][1:seq_len-1].cpu().numpy()
#         # mean_pool = np.mean(seq_embedding, axis=0)

#         # embeddings.append(mean_pool)

#         sequence = ' '.join(sequence)
#         sequence = sequence.replace('U', 'X')
#         sequence = sequence.replace('O', 'X')
#         sequence = sequence.replace('B', 'X')
#         sequence = sequence.replace('Z', 'X')

#         tokenized_sequence = tokenizer.encode_plus(sequence, add_special_tokens=True, max_length=20000, truncation=True)
#         input_ids = torch.tensor([tokenized_sequence['input_ids']]).to(device)
#         attention_mask = torch.tensor([tokenized_sequence['attention_mask']]).to(device)

#         with torch.no_grad():
#             last_hidden_states = model(input_ids, attention_mask=attention_mask)[0]

#         embedding = last_hidden_states[0].cpu().numpy()
#         seq_len = (attention_mask[0] == 1).sum()
#         seq_embedding = embedding[1:seq_len-1]
#         mean_pool = np.mean(seq_embedding, axis=0)

#         embeddings.append(mean_pool)

#     return np.array(embeddings)




# # Extract embeddings
# embeddings = extract_embeddings(model, tokenizer, sequences)
# print(embeddings.shape)

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

# print(X_train.shape)  # Expected: (n_samples, 1024)
# print(X_test.shape)   # Expected: (n_samples, 1024)

# # Reshape X_train and X_test correctly to 2D arrays
# X_train = X_train.reshape(X_train.shape[0], -1)
# X_test = X_test.reshape(X_test.shape[0], -1)

# print(X_train.shape)  # Expected: (n_samples, 1024)
# print(X_test.shape)   # Expected: (n_samples, 1024)

# # Train logistic regression
# lr = LogisticRegression(max_iter=1000)
# lr.fit(X_train, y_train)

# # Evaluate
# y_pred = lr.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy: {accuracy:.4f}')

# # Save the model
# joblib.dump(lr, 'terpene_lr_model.pkl')
