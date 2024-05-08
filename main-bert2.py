import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re

import argparse
parser = argparse.ArgumentParser(description='')
parser.add_argument('-epochs', type=int, default=3, help='epochs')
args = parser.parse_args()
epochs = args.epochs

def preprocess_text(text):
    text = text.lower()  # 轉換為小寫
    text = re.sub(r'\d+', '', text)  # 移除數字
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # 移除標點符號
    return text

# 加載數據
train_data = pd.read_json('data/train.json')
train_data['text'] = train_data['text'].apply(preprocess_text)
test_data = pd.read_json('data/test.json')
test_data['text'] = test_data['text'].apply(preprocess_text)

# 使用 BERT 的 Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 將文本轉換為 BERT 的輸入格式
def encode_data(data):
    input_ids = []
    attention_masks = []
    for text in data:
        encoded = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',  # 更新填充方式
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

X_train_ids, X_train_masks = encode_data(train_data['text'])
Y_train = torch.tensor(train_data['rating'].values - 1)  # 整數標籤，從0開始

# 分割數據為訓練集和驗證集
X_train, X_val, X_mask_train, X_mask_val, Y_train, Y_val = train_test_split(X_train_ids, X_train_masks, Y_train, test_size=0.1, random_state=42)

# 創建 DataLoader
batch_size = 8
train_data = TensorDataset(X_train, X_mask_train, Y_train)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
val_data = TensorDataset(X_val, X_mask_val, Y_val)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

# 加載 BERT 模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
model.cuda()  # 強制使用 GPU

# 設定優化器
optimizer = AdamW(model.parameters(), lr=2e-5)

# 訓練模型
model.train()
for epoch in range(epochs):  # 設定迴圈次數
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        batch = [r.cuda() for r in batch]
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        model.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss {total_loss / len(train_dataloader)}')

# 評估模型在驗證集上的表現
model.eval()
predictions, true_vals = [], []
for batch in val_dataloader:
    batch = [t.cuda() for t in batch]
    inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predictions.append(logits.detach().cpu().numpy())
    true_vals.append(batch[2].detach().cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
predictions = np.argmax(predictions, axis=1)
true_vals = np.concatenate(true_vals)
accuracy = accuracy_score(true_vals, predictions)
print(f'Accuracy on validation set: {accuracy}')

# 使用模型對測試集進行預測
test_ids, test_masks = encode_data(test_data['text'])
test_data = TensorDataset(test_ids, test_masks)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

model.eval()
test_predictions = []
for batch in test_dataloader:
    batch = [t.cuda() for t in batch]
    inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    test_predictions.append(logits.detach().cpu().numpy())

test_predictions = np.concatenate(test_predictions, axis=0)
test_predictions = np.argmax(test_predictions, axis=1) + 1  # 加1是因為索引從0開始，而評級從1開始

# 創建提交文件的 DataFrame
submission = pd.DataFrame({
    'index': ['index_' + str(i) for i in range(0, len(test_predictions))],
    'rating': [str(rating) + '.0' for rating in test_predictions]
})

# 將 DataFrame 保存為 CSV 文件，不包含索引列
submission.to_csv('output_bert/submission'+str(int(accuracy * 10000))+'_'+str(epochs)+'.csv', index=False)
print("提交文件已保存。")
