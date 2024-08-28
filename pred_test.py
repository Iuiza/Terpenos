import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

import argparse
from Bio import SeqIO
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
import joblib

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('input_file', type=str, help='Input FASTA file')
parser.add_argument('output_file', type=str, help='Output txt file with predicted labels')

args = parser.parse_args()

# Check if the input file is a FASTA file.
with open(args.input_file, 'r') as f:
    first_line = f.readline()
    if not first_line.startswith('>'):
        raise ValueError('Input file is not a FASTA file.')

# Process the input file and write the output to the output file.
with open(args.input_file, 'r') as f:
    records = list(SeqIO.parse(f, 'fasta'))
    sequences_ids = [(str(record.seq), str(record.id)) for record in records]

# Load the ProtBert model and tokenizer
print('Loading ProtBert model and tokenizer...')
# tokenizer = BertTokenizer.from_pretrained('./terpene_model')
tokenizer = BertTokenizer.from_pretrained('rostlab/prot_bert_bfd', do_lower_case=False)
model = BertModel.from_pretrained('./terpene_model')
model.to(device)

# Load the logistic regression model trained for terpene classification.
print('Loading logistic regression model...')
lr = joblib.load('terpene_lr_model.pkl')

# For each sequence, tokenize it and pass it through the ProtBert model.
predictions = []
print("Sequence ID\t\tPredicted label")
print("------------\t\t---------------")
for sequence, id in sequences_ids:
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

    # Predict the label
    prediction = lr.predict([mean_pool])
    predictions.append(f"Sequence:{id}\tPrediction:{prediction[0]}")

    # Print the id and the prediction
    print(f"{id}\t{prediction[0]}")

# Write the output to the output file
with open(args.output_file, 'w') as f:
    for prediction in predictions:
        f.write(prediction + '\n')

print('Finished.')