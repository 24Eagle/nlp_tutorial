"""
#@time：2026/10/17  13:26
#@author：LiuYang
"""
import torch
from torch import nn
import config

class PositionEncoding(nn.Module):
    """
        生成位置编码矩阵(L,E)
    """
    def __init__(self, d_model, max_len=500):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        pos = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1) # pos.shape: (max_len, 1)
        _2i = torch.arange(0, self.d_model, step=2, dtype=torch.float) # _2i.shape: (d_model/2,)
        div_term = torch.pow(10000, _2i / self.d_model)
        sins = torch.sin(pos / div_term) # sins.shape: (max_len, d_model/2)
        coss = torch.cos(pos / div_term) # coss.shape: (max_len, d_model/2)

        pe = torch.zeros(self.max_len, self.d_model, dtype=torch.float) # pe.shape: (max_len, d_model)

        pe[:, 0::2] = sins
        pe[:, 1::2] = coss

        self.register_buffer('pe', pe)

    def forward(self, x):
        """

        :param x: (N,L,E)
        :return:
        """

        seq_len = x.size(1)
        return x + self.pe[:seq_len]

class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_index, en_padding_index):
        super().__init__()
        # 嵌入层和位置编码
        # 编码器的
        self.src_embedding = nn.Embedding(num_embeddings=zh_vocab_size, embedding_dim=config.DIM_MODEL,
        padding_idx=zh_padding_index)
        # 解码器的
        self.tgt_embedding = nn.Embedding(num_embeddings=en_vocab_size, embedding_dim=config.DIM_MODEL,
        padding_idx=en_padding_index)
        # 位置编码，两处复用
        self.position_encoding = PositionEncoding(d_model=config.DIM_MODEL)
        # Transformer层
        self.transformer = nn.Transformer(d_model=config.DIM_MODEL,
        nhead=config.NUM_HEADS,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        batch_first=True)
        # 输出线性层
        self.linear = nn.Linear(config.DIM_MODEL, en_vocab_size)

    def encode(self, src, src_pad_mask):
        """

        :param src: (N,S=src_len)
        :param src_pad_mask: (N,S)
        :return:
        """
        # 1.词嵌入
        src_embed = self.src_embedding(src)
        # embed：(N,S,E=d_model)
        src_embed = self.position_encoding(src_embed)
        # (N,S,E)
        memory = self.transformer.encoder(src=src_embed, src_key_padding_mask=src_pad_mask)
        return memory

    def decode(self, tgt, memory, tgt_mask, tgt_pad_mask, src_pad_mask):
        """

        :param tgt: (N,T)
        :param memory:
        :param tgt_mask: (T,T)
        :param tgt_pad_mask:
        :param src_pad_mask:
        :return:
        """
        tgt_embed = self.tgt_embedding(tgt)
        tgt_embed = self.position_encoding(tgt_embed)

        output = self.transformer.decoder(tgt=tgt_embed, memory=memory, tgt_mask=tgt_mask,
        tgt_key_padding_mask=tgt_pad_mask,
        memory_key_padding_mask=src_pad_mask)
        return self.linear(output)

    def forward(self, src, tgt, src_pad_mask, tgt_pad_mask, tgt_mask):
        """
        输入源序列（N,S）,目标序列（N,T）
        这里的前向传播是针对整个模型的，而不是对于每一个编码器或者解码器里面的前向传播层
        :param src:
        :param tgt:
        :param src_pad_mask:
        :param tgt_pad_mask:
        :param tgt_mask:
        :return:
        """
        # 编码
        memory = self.encode(src, src_pad_mask)
        # 解码
        output = self.decode(tgt, memory, tgt_mask, tgt_pad_mask, src_pad_mask)
        return output