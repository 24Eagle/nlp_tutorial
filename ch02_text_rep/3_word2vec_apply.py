"""
#@time：2026/10/14  15:35
#@author：LiuYang
"""
import torch
import torch.nn as nn
import jieba

from gensim.models import KeyedVectors  # 加载词向量文件

# 1.加载预训练的词向量
wv_model = KeyedVectors.load_word2vec_format("./data/word2vec.kv")

# 获取词向量维度
vector_dim = wv_model.vector_size
print("词向量维度:", vector_dim)

# 增加一个处理，就是对于后续word2id出现未登录词（oov）的处理
unk_token = '<unk>'
id2word = [unk_token]+wv_model.index_to_key
word2id = {word : id for id, word in enumerate(id2word)}
# 2.构建词表获取大小
# word2id = wv_model.key_to_index
vocab_size = len(word2id)
print("词表大小:", vocab_size)

# 3.构造一个词向量矩阵 tensor
embedding_matrix = torch.zeros(vocab_size, vector_dim)
for word, id in word2id.items():
    # wv_model里面没有未登录词
    if word in wv_model:
        embedding_matrix[id] = torch.tensor(wv_model[word])
print(embedding_matrix.shape)
# 4.构建神经网络的嵌入层,做微调就是freeze是False
embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
# embedding = nn.Embedding.from_pretrained(torch.tensor(wv_model.vectors), freeze=True)

# 5.测试
text = "我喜欢乘坐宇宙飞船"
# 5.1 分词
tokens = jieba.lcut(text)
# print(tokens)

# 5.2 构架词表，前面已经构建了，所以就将该语料id化
# ids = [word2id[word] for word in tokens]
# 当出现未登录词时，就把其转化为unk对应的那个id
# get就是获取到就是第一个位置，获取不到就是第二个位置
ids = [word2id.get(word, word2id[unk_token]) for word in tokens]
# print(ids)
# 5.3 构建神经网络输入数据(tensor)
input = torch.tensor([ids])
# 5.4 前向传播，查找词向量
output = embedding(input)

print(output)
print(output.shape)