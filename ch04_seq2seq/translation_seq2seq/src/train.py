"""
#@time：2026/10/17  13:26
#@author：LiuYang
"""
import torch
from torch import nn, optim
from config import *
from dataset import get_dataloader  # 获取数据加载器
from model import TranslationModel  # 模型
from torch.utils.tensorboard import SummaryWriter  # 写入器
# 加进度条
from tqdm import tqdm
import time
from tokenizer import ChineseTokenizer, EnglishTokenizer


# 定义引擎函数，训练一个epoch，返回平均损失
def train_one_epoch(model, train_loader, loss, optimizer, device):
    # model.train()
    total_loss = 0
    # 按批次进行迭代
    for inputs, targets in tqdm(train_loader, desc="训练："):
        inputs, targets = inputs.to(device), targets.to(device)  # 形状都是（N,L）
        # 前向传播
        # 分阶段
        # 1.1编码阶段,得到一批上下文向量（N,hidden_size）
        context_vectors = model.encoder(inputs)
        # 1.2解码（teacher forcing）阶段，基于目标序列得到解码的输入和目标
        decoder_inputs = targets[:, :-1]
        decoder_targets = targets[:, 1:].long()  # (N,L)
        # 用编码得到的上下文向量得到的初始隐状态,(1,N,hidden_size)
        decoder_h0 = context_vectors.unsqueeze(0)
        # 解码器前向传播,(N,L,vocab_size)
        print(decoder_inputs.dtype, decoder_inputs.shape)
        # 注意输入的数据类型
        decoder_outputs, _ = model.decoder(decoder_inputs.long(), decoder_h0)
        # 计算损失
        loss_value = loss(decoder_outputs.transpose(1, 2), decoder_targets)
        # 反向传播
        loss_value.backward()
        # 更新参数
        optimizer.step()
        # 梯度清零
        optimizer.zero_grad()
        # 累加损失
        total_loss += loss_value.item()
    return total_loss / len(train_loader)


# 训练整体流程
def train():
    # 1.定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 2.定义数据加载器
    train_loader = get_dataloader()
    # 3.获取词表,创建分词器
    cn_tokenizer = ChineseTokenizer.from_vocab(MODEL_DIR / CN_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(MODEL_DIR / EN_VOCAB_FILE)

    # 4.定义模型
    model = TranslationModel(cn_tokenizer.vocab_size, en_tokenizer.vocab_size, cn_tokenizer.pad_id,
                             en_tokenizer.pad_id).to(device)
    # 5.定义损失函数
    loss = nn.CrossEntropyLoss(ignore_index=en_tokenizer.pad_id)
    # 6.定义优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 定义一个tensorboard写入器
    writer = SummaryWriter(log_dir=LOG_DIR / time.strftime('%Y-%m-%d_%H-%M-%S'))
    # 7.核心训练流程,按epoch进行迭代
    min_loss = float('inf')  # 记录最小训练损失
    for epoch in range(EPOCHS):
        print("=" * 10, f"EPOCH:{epoch + 1}", "=" * 10)
        this_loss = train_one_epoch(model, train_loader, loss, optimizer, device)
        print("本轮训练损失:", this_loss)
        # 将损失写入日志
        writer.add_scalar('loss', this_loss, epoch + 1)

        # 判断损失是否下降，保存最优模型
        if this_loss < min_loss:
            min_loss = this_loss
            torch.save(model.state_dict(), MODEL_DIR / BEST_MODEL)
            print("模型保存成功！")
    writer.close()


if __name__ == "__main__":
    train()
