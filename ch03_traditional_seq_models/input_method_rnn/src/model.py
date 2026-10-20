"""
#@time：2026/10/17  13:26
#@author：LiuYang
"""
import torch
import torch.nn as nn

from config import *


# 自定义神经网络类（RNNLM）
class InputMethodRNNModel(nn.Module):
    # 初始化
    def __init__(self, vocab_size):
        super().__init__()
        # 词嵌入层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE)
        # 单层单向rnn
        self.rnn = nn.RNN(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
        # 全连接层
        self.linear = nn.Linear(in_features=HIDDEN_SIZE, out_features=vocab_size)

    def forward(self, x):
        # 词嵌入
        embed = self.embedding(x)
        # print(len(x))
        # Rnn前向传播，得到输出(N,L,hidden_size)
        output, _ = self.rnn(embed)
        # 取最后一个时间步的隐状态，作为上下文特征向量
        feature = output[:, -1, :]
        # 经过全连接层整合特征，  得到多分类输出
        result = self.linear(feature)
        return result


if __name__ == "__main__":
    vocab_size = 1000
    input = torch.randint(vocab_size, size=(64, 5))
    model = InputMethodRNNModel(vocab_size)
    output = model(input)
    print(output.shape)
