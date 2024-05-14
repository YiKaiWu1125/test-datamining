import json
import matplotlib.pyplot as plt

# 載入JSON文件
with open('test.json', 'r') as file:
    data = json.load(file)

# 提取評論文字長度
text_lengths = [len(review['text']) for review in data]

# 繪製評論文字長度分佈圖
plt.figure(figsize=(5, 6))
plt.hist(text_lengths, bins=30, edgecolor='black')
plt.title('Comment text length distribution')
plt.xlabel('text length')
plt.ylabel('frequency')
plt.grid(True)
plt.show()

# 列出一些具體的評論文字長度
for i, length in enumerate(text_lengths[:5]):
    print(f'評論{i+1}：{length} 字符')
