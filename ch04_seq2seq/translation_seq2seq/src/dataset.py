"""
#@time：2026/10/17  13:26
#@author：LiuYang
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from config import *
from torch.nn.utils.rnn import pad_sequence


# 自定义数据集类
class TranslationDataset(Dataset):
    # 初始化
    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    # 获取长度
    def __len__(self):
        return len(self.data)

    # 根据index获取元素
    def __getitem__(self, index):
        input = torch.tensor(self.data[index]['cn'], dtype=torch.long)
        target = torch.tensor(self.data[index]['en'], dtype=torch.float)
        return input, target


def collate_fn(batch):
    """
    整理函数，将一批数据长度对齐（填充）
    batch：形如((input0,target0),(input1,target1)...)
    :return:
    """
    # 因为dataset返回值是一个元组（input,target），先将其分为两个列表
    input_tensor_list = [item[0] for item in batch]
    target_tensor_list = [item[1] for item in batch]
    # 合并成长度对齐的一个batch tensor
    # 这里设置batch_first为True的原因是，默认的天聪方向是把每一个序列按照最长的来对其，然后一列一列的放，设置之后
    # 就是把每个序列填充后一行一行放来对其
    input_batch_tenor = pad_sequence(input_tensor_list, batch_first=True, padding_value=0)
    target_batch_tenor = pad_sequence(target_tensor_list, batch_first=True, padding_value=0)

    return input_batch_tenor, target_batch_tenor


# 获取DataLoader的函数
def get_dataloader(train=True):
    path = PROCESSED_DATA_DIR / (TRAIN_DATA_FILE if train else TEST_DATA_FILE)
    dataset = TranslationDataset(path)
    dataloder = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    return dataloder


if __name__ == "__main__":
    train_dataloader = get_dataloader(train=True)
    test_dataloader = get_dataloader(train=False)
    for input, target in train_dataloader:
        print(input.shape, target.shape)
        break
