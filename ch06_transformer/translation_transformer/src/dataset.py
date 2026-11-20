"""
#@time：2026/10/17  13:26
#@author：LiuYang
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config

class TranslationDataset(Dataset):
    """
    翻译数据集，加载已编码的中英文索引。
    """

    def __init__(self, data_path):
        """
        初始化数据集。
        :param data_path: 数据文件路径（JSONL 格式）。
        """
        self.data = pd.read_json(data_path, lines=True).to_dict(orient='records')

    def __len__(self):
        """
        数据集样本数。
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        获取指定样本。

        :param index: 样本索引。
        :return: (input_tensor, target_tensor)
        """
        input_tensor = torch.tensor(self.data[index]['zh'], dtype=torch.long)

        target_tensor = torch.tensor(self.data[index]['en'], dtype=torch.long)
        return input_tensor, target_tensor

def get_dataloader(train=True):
    """
    构建数据加载器。

    :param train: 是否加载训练集。
    :return: DataLoader 实例。
    """
    data_path = config.PROCESSED_DATA_DIR / ('indexed_train.jsonl' if train else 'indexed_test.jsonl')
    dataset = TranslationDataset(data_path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

if __name__ == '__main__':
    train_loader = get_dataloader(train=True)
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)
        break