import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel

# 加載數據
train_data = pd.read_json('data/train.json')
test_data = pd.read_json('data/test.json')

# 設置 BERT 預訓練模型
PRETRAINED_MODEL_NAME = 'bert-base-chinese'  # 使用中文模型

# 加載 BERT 分詞器
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# 文本預處理函數
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.encode_plus(text, max_length=max_len, truncation=True,
                                     padding='max_length', add_special_tokens=True)
        all_tokens.append(text['input_ids'])
        all_masks.append(text['attention_mask'])
        all_segments.append(text['token_type_ids'])
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

# 序列化訓練和測試數據
MAX_LENGTH = 250  # 適當調整長度以滿足實際需求
train_inputs, train_masks, train_segments = bert_encode(train_data['text'].values, tokenizer, max_len=MAX_LENGTH)
test_inputs, test_masks, test_segments = bert_encode(test_data['text'].values, tokenizer, max_len=MAX_LENGTH)

# 將標籤轉換為分類格式
Y = pd.get_dummies(train_data['rating']).values

# 分割數據為訓練集和驗證集
X_train, X_val, Y_train, Y_val = train_test_split(train_inputs, Y, test_size=0.2, random_state=42)
train_masks, val_masks, _, _ = train_test_split(train_masks, Y, test_size=0.2, random_state=42)

# 構建 BERT 模型
input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
input_mask = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_mask")

bert_model = TFBertModel.from_pretrained(PRETRAINED_MODEL_NAME)
bert_output = bert_model(input_ids, attention_mask=input_mask)[1]

dense_layer = tf.keras.layers.Dense(5, activation='softmax')(bert_output)

model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=dense_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 訓練模型
epochs = 3  # BERT 往往不需要太多輪次
batch_size = 16  # 根據 GPU 記憶體調整
history = model.fit(
    [X_train, train_masks], Y_train, 
    validation_data=([X_val, val_masks], Y_val),
    epochs=epochs, batch_size=batch_size
)

# 評估模型
score, acc = model.evaluate([X_val, val_masks], Y_val, verbose=2)
print(f"模型在驗證集上的準確率: {acc * 100:.2f}%")

# 使用模型對測試集進行預測
predictions = model.predict([test_inputs, test_masks])
predicted_ratings = np.argmax(predictions, axis=1) + 1  # 索引從0開始，而評級從1開始

# 創建提交文件的 DataFrame
submission = pd.DataFrame({
    'index': 'index_'+np.arange(1, len(predicted_ratings) + 1),
    'rating': predicted_ratings+'0'
})

# 將 DataFrame 保存為 CSV 文件，不包含索引列
submission.to_csv('submission.csv', index=False)
print("提交文件已保存為 'submission.csv'。")
