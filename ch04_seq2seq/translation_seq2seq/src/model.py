"""
#@time：2026/10/17  13:26
#@author：LiuYang
"""
import torch
import torch.nn as nn

from config import *


# 自定义编码器（基于GRU）
class TranslationEncoder(nn.Module):
    # 初始化
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        # 词嵌入层,指定填充词id
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx=padding_idx)
        # 单层单向GRU
        self.gru = nn.GRU(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)

    def forward(self, x):
        # 词嵌入
        embed = self.embedding(x)
        # print(len(x))

        output, _ = self.gru(embed)
        # 用列表索引取真实的最后一个时间步的隐状态，作为上下文特征向量
        indices = torch.arange(output.shape[0])  # 行号
        lengths = (x != self.embedding.padding_idx).sum(dim=1)  # 计算每个序列的真实长度，N是第0维，L是第1维
        features = output[indices, lengths - 1]

        return features


# 自定义解码器
class TranslationDecoder(nn.Module):
    # 初始化
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        # 词嵌入层,指定填充词id
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE, padding_idx=padding_idx)
        # 单层单向GRU
        self.gru = nn.GRU(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
        # 全连接层
        self.linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=vocab_size)

    def forward(self, x, h0=None):
        # 词嵌入
        embed = self.embedding(x)
        # 通过GRU得到当前的输出隐状态(特征向量)
        output, hn = self.gru(embed, h0)
        # 整合特征，预测分类
        output = self.linear(output)

        return output, hn


# 自定义Seq2Seq模型
class TranslationModel(nn.Module):
    def __init__(self, cn_vocab_size, en_vocab_size, en_padding_idx, cn_padding_idx):
        super().__init__()
        self.encoder = TranslationEncoder(cn_vocab_size, cn_padding_idx)
        self.decoder = TranslationDecoder(en_vocab_size, en_padding_idx)


if __name__ == "__main__":
    model = TranslationModel(cn_vocab_size=1000, en_vocab_size=1024, cn_padding_idx=0, en_padding_idx=0)
    print(model.encoder)
    print(model.decoder)