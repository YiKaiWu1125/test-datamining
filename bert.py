import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
import pandas as pd
from torch.optim import AdamW
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import re
import argparse
import time

# 命令行参数解析
parser = argparse.ArgumentParser(description='')
parser.add_argument('-epochs', type=int, default=3, help='epochs')
parser.add_argument('-batch_size', type=int, default=8, help='batch_size')
parser.add_argument('-weight_decay', type=float, default=0.01, help='weight_decay')
args = parser.parse_args()

epochs = args.epochs
batch_size = args.batch_size
weight_decay = args.weight_decay

def preprocess_text(text):
    text = text.lower()  # 轉換為小寫
    text = re.sub(r'\d+', '', text)  # 移除數字
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # 移除標點符號
    return text

# 加載數據
train_data = pd.read_json('data/train.json')
train_data['text'] = (train_data['title'] + ' ')*5 + train_data['text'].apply(preprocess_text)
test_data = pd.read_json('data/test.json')
test_data['text'] = (test_data['title'] + ' ')*5 + test_data['text'].apply(preprocess_text)

# 使用 BERT 的 Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', force_download=False)

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
Y_train = torch.tensor(train_data['rating'].values - 1)

# 切分訓練集與驗證集
X_train_ids, X_val_ids, X_train_masks, X_val_masks, Y_train, Y_val = train_test_split(X_train_ids, X_train_masks, Y_train, test_size=0.1, random_state=42)

train_data = TensorDataset(X_train_ids, X_train_masks, Y_train)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

val_data = TensorDataset(X_val_ids, X_val_masks, Y_val)
val_dataloader = DataLoader(val_data, batch_size=batch_size)

# 加載模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5)
model.cuda()

# 設置優化器和學習率衰減
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=weight_decay)
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 訓練模型
best_accuracy = 0
for epoch in range(epochs):
    model.train()
    total_loss = 0
    start_time = time.time()
    correct_predictions = 0
    for batch in train_dataloader:
        batch = [t.cuda() for t in batch]  # 將數據移動到GPU
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        correct_predictions += (preds == batch[2]).sum().item()
        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  # 更新學習率
        torch.cuda.empty_cache()
    train_acc = correct_predictions / len(train_data)
    train_duration = time.time() - start_time

    # 驗證模型在驗證集上的表現
    model.eval()
    val_correct_predictions = 0
    val_start_time = time.time()
    for batch in val_dataloader:
        batch = [t.cuda() for t in batch]
        inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[2]}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        val_correct_predictions += (preds == batch[2]).sum().item()
    val_acc = val_correct_predictions / len(val_data)
    val_duration = time.time() - val_start_time
    print(f'Epoch {epoch + 1}, Loss {total_loss / len(train_dataloader)}, Training Accuracy: {train_acc}, Time: {train_duration:.2f}s, Validation Accuracy: {val_acc}, Val Time: {val_duration:.2f}s')

    # 保存最佳模型
    if val_acc > best_accuracy:
        best_accuracy = val_acc
        index = epoch
        torch.save(model.state_dict(), 'best_model.pth')

# 將測試數據編碼為BERT輸入格式
test_ids, test_masks = encode_data(test_data['text'])
test_labels = torch.tensor([0]*len(test_ids))  # 假設測試集沒有標籤，我們只需輸入一個占位符

# 創建測試數據的TensorDataset對象
test_dataset = TensorDataset(test_ids, test_masks, test_labels)

# 創建DataLoader來迭代測試數據
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# 使用最佳模型進行預測
model.load_state_dict(torch.load('best_model.pth'))
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
test_predictions = np.argmax(test_predictions, axis=1) + 1  # 加1是因为索引从0开始，而评级从1开始

# 创建提交文件的 DataFrame
submission = pd.DataFrame({
    'index': ['index_' + str(i) for i in range(0, len(test_predictions))],
    'rating': [str(rating) + '.0' for rating in test_predictions]
})

# 将 DataFrame 保存为 CSV 文件，不包含索引列
submission.to_csv('output_new/VAcc'+str(int(best_accuracy * 10000))+'_EpochsSize'+str(epochs)+'_BS'+str(batch_size)+'_WD'+str(weight_decay)+'_Bestecho'+str(index)+'.csv', index=False)
print("提交文件已保存。")
