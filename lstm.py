import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow as tf

# 加载数据
train_data = pd.read_json('data/train.json')
test_data = pd.read_json('data/test.json')

# 查看数据的前几行，确保数据被正确加载
print(train_data.head())

# 设置最大的词汇量和序列长度
MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100

# 初始化和配置Tokenizer
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train_data['text'].values)

# 将文本转换为序列并进行填充，以确保相同长度
X = tokenizer.texts_to_sequences(train_data['text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)

# 将标签转换为分类格式
Y = pd.get_dummies(train_data['rating']).values

# 分割数据为训练集和验证集
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# 构建模型

with tf.device('/gpu:0'):  # 强制使用第一个GPU
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
    model.add(SpatialDropout1D(0.2))
    # 移除 recurrent_dropout 并保持默认激活函数为 tanh
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # 显示模型摘要
    model.summary()

    # 训练模型
    epochs = 50
    batch_size = 64
    history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, Y_val))

    # 评估模型在验证集上的表现
    score, acc = model.evaluate(X_val, Y_val, verbose=2)
    print(f"模型在验证集上的准确率: {acc * 100:.2f}%")

    # 使用模型对测试集进行预测
    test_sequences = tokenizer.texts_to_sequences(test_data['text'].values)
    test_padded = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    predictions = model.predict(test_padded)

    # 将预测结果转换为整数评级
    predicted_ratings = np.argmax(predictions, axis=1) + 1  # 加1是因为索引从0开始，而评级从1开始

    # 创建提交文件的 DataFrame
    submission = pd.DataFrame({
        'index': np.arange(1, len(predicted_ratings) + 1),  # 创建一个从1开始的索引
        'rating': predicted_ratings
    })

    # 将 DataFrame 保存为 CSV 文件，不包含索引列
    submission.to_csv('submission.csv', index=False)
    print("提交文件已保存为 'submission.csv'。")
