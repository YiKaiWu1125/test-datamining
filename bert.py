import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertModel

# 加载数据
train_data = pd.read_json('data/train.json')
test_data = pd.read_json('data/test.json')

# 设置 BERT 预训练模型
PRETRAINED_MODEL_NAME = 'bert-base-chinese'  # 以中文模型为例，根据需求更改

# 加载 BERT 分词器
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)

# 文本预处理函数
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

# 序列化训练和测试数据
MAX_LENGTH = 250  # 适当调整长度以满足实际需求
train_inputs, train_masks, train_segments = bert_encode(train_data['text'].values, tokenizer, max_len=MAX_LENGTH)
test_inputs, test_masks, test_segments = bert_encode(test_data['text'].values, tokenizer, max_len=MAX_LENGTH)

# 将标签转换为分类格式
Y = pd.get_dummies(train_data['rating']).values

# 分割数据为训练集和验证集
X_train, X_val, Y_train, Y_val = train_test_split(train_inputs, Y, test_size=0.2, random_state=42)
train_masks, val_masks, _, _ = train_test_split(train_masks, Y, test_size=0.2, random_state=42)

# 构建 BERT 模型
input_ids = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_ids")
input_mask = tf.keras.layers.Input(shape=(MAX_LENGTH,), dtype=tf.int32, name="input_mask")

bert_model = TFBertModel.from_pretrained(PRETRAINED_MODEL_NAME)
bert_output = bert_model(input_ids, attention_mask=input_mask)[1]

dense_layer = tf.keras.layers.Dense(5, activation='softmax')(bert_output)

model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=dense_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 训练模型
epochs = 3  # BERT 往往不需要太多轮次
batch_size = 16  # 根据 GPU 内存调整
history = model.fit(
    [X_train, train_masks], Y_train, 
    validation_data=([X_val, val_masks], Y_val),
    epochs=epochs, batch_size=batch_size
)

# 评估模型
score, acc = model.evaluate([X_val, val_masks], Y_val, verbose=2)
print(f"模型在验证集上的准确率: {acc * 100:.2f}%")

# 使用模型对测试集进行预测
predictions = model.predict([test_inputs, test_masks])
predicted_ratings = np.argmax(predictions, axis=1) + 1  # 索引从0开始，而评级从1开始

# 创建提交文件的 DataFrame
submission = pd.DataFrame({
    'index': np.arange(1, len(predicted_ratings) + 1),
    'rating': predicted_ratings
})

# 将 DataFrame 保存为 CSV 文件，不包含索引列
submission.to_csv('submission.csv', index=False)
print("提交文件已保存为 'submission.csv'。")
