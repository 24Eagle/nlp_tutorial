"""
#@time：2026/10/17  13:26
#@author：LiuYang
"""
import torch
import torch.nn as nn

from config import *


class Attention(nn.Module):
    def forward(self, decoder_hiddens, encoder_outputs):
        # 1.计算注意力评分
        # 解码器隐状态(N,L,hidden_size)和编码器的输出隐状态(N,L_enc,hidden_size)
        attention_scores = torch.bmm(decoder_hiddens, encoder_outputs.transpose(1, 2))
        # 评分形状（N,L,L_enc)
        # 2.计算注意力权重,dim=-1表示对L_enc进行评分
        attention_weights = torch.softmax(attention_scores, dim=-1)
        # 3.计算上下文向量（N,L,L_enc)*(N,L_enc,hidden_size)=(N,L,hidden_size)
        content_vector = torch.bmm(attention_weights, encoder_outputs)
        return content_vector


# 自定义编码器（基于GRU）
class TranslationEncoder(nn.Module):
    # 初始化
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        # 词嵌入层,指定填充词id
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE,
                                      padding_idx=padding_idx)
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

        return output, features


# 自定义解码器
class TranslationDecoder(nn.Module):
    # 初始化
    def __init__(self, vocab_size, padding_idx):
        super().__init__()
        # 词嵌入层,指定填充词id
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=EMBEDDING_SIZE,
                                      padding_idx=padding_idx)
        # 单层单向GRU
        self.gru = nn.GRU(input_size=EMBEDDING_SIZE, hidden_size=HIDDEN_SIZE, batch_first=True)
        # Attention层
        self.attention = Attention()
        # 全连接层
        self.linear = nn.Linear(in_features=2 * HIDDEN_SIZE, out_features=vocab_size)

    def forward(self, x, h0, encoder_ouputs):
        # 词嵌入
        embed = self.embedding(x)
        # 通过GRU得到当前的输出隐状态(特征向量)
        output, hn = self.gru(embed, h0)
        # 增加注意力机制，基于解码器隐状态(N,L,hidden_size)和编码器的输出隐状态(N,L_enc,hidden_size)，计算动态上下文向量
        context_vector = self.attention(output, encoder_ouputs)
        # 上下文向量形状(N,L,hidden_size)
        # 将动态上下文向量拼接融合到隐状态信息中
        combined = torch.cat([context_vector, output], dim=-1)
        # 合并特征向量(N,L,hidden_size*2)

        # 整合特征，预测分类
        output = self.linear(combined)

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
