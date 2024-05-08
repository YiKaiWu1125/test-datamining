import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import classification_report
import pandas as pd

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
train_data = pd.read_json('data/train.json')
test_data = pd.read_json('data/test.json')

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def encode_data(data):
    input_ids = []
    attention_masks = []

    for sent in data:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=32,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    return input_ids, attention_masks

# Encode data
train_inputs, train_masks = encode_data(train_data['text'].values)
train_labels = torch.tensor(train_data['rating'].values - 1)
test_inputs, test_masks = encode_data(test_data['text'].values)

# Create DataLoader for training set
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=16)

# Initialize BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=5,
    output_attentions=False,
    output_hidden_states=False,
)
model.to(device)

# Set optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)

# Training loop
model.train()
for epoch_i in range(3):
    print(f"Epoch {epoch_i+1} of 3")
    total_train_loss = 0

    for step, batch in enumerate(train_dataloader):
        b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
        model.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs[0]
        total_train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if step % 20 == 0 and step != 0:
            print(f"  Batch {step} of {len(train_dataloader)}. Loss: {loss.item()}.")

    # Validation
    model.eval()
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        for val_batch in val_dataloader:
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in val_batch)
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            val_predictions.extend(logits.argmax(dim=-1).cpu().numpy())
            val_labels.extend(b_labels.cpu().numpy())
    
    val_accuracy = classification_report(val_labels, val_predictions, output_dict=True)['accuracy']
    print(f"  Validation Accuracy: {val_accuracy}")

# Testing
model.eval()
with torch.no_grad():
    predictions = model(test_inputs.to(device), token_type_ids=None, attention_mask=test_masks.to(device))
predicted_indices = predictions[0].argmax(dim=-1).cpu().numpy()
predicted_ratings = predicted_indices + 1

# Output results
submission = pd.DataFrame({
    'index': ['index_' + str(i) for i in range(len(predicted_ratings))],
    'rating': [str(rating) + '.0' for rating in predicted_ratings]
})
submission.to_csv('output/submission.csv', index=False)
print("Submission file saved as 'output/submission.csv'.")
