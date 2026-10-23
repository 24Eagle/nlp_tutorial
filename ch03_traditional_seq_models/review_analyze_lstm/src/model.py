"""
#@time：2026/10/17  13:26
#@author：LiuYang
"""
import torch
import torch.nn as nn

from config import *


# 自定义神经网络类（LSTM）
class ReviewAnalysisModel(nn.Module):
    # 初始化
    def __init__(self, vocab_size,padding_idx):
        super().__init__()
        # 词嵌入层,指定填充词id
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE,padding_idx=padding_idx)
        # 单层单向rnn
        self.lstm = nn.LSTM(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
        # 全连接层,最终输出维度为1，代表属于正向评价的评分
        self.linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=1)

    def forward(self, x):
        # 词嵌入
        embed = self.embedding(x)
        # print(len(x))
        # LSTM前向传播，得到输出(N,L,hidden_size)
        output, (hn, cn) = self.lstm(embed)
        # 用列表索引取真实的最后一个时间步的隐状态，作为上下文特征向量
        indices = torch.arange(output.shape[0])  # 行号
        lengths = (x != self.embedding.padding_idx).sum(dim=1)  # 计算每个序列的真实长度，N是第0维，L是第1维
        features = output[indices, lengths-1]
        # 经过全连接层整合特征，得到预测输出（N,1）,在降维成一维（N,）
        result = self.linear(features).squeeze(-1)
        return result


if __name__ == "__main__":
    vocab_size = 1000
    input = torch.randint(vocab_size, size=(64, 5))
    model = ReviewAnalysisModel(vocab_size, padding_idx=0)
    output = model(input)
    print(output.shape)
