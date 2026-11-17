"""
#@time：2026/10/17  13:26
#@author：LiuYang
"""
import jieba
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tokenizer import ChineseTokenizer, EnglishTokenizer
from config import *


# 构建数据集的函数，传入原始语料和词表word2id,返回{‘input’：{ids},'target':id}
def preprocess():
    print("============开始数据预处理==============")
    # 1.读取csv文件,并提取两列，去除缺省值,包含了第0列和第1列，分别取名字为en和cn
    df = pd.read_csv(RAW_DATA_DIR / RAW_DATA_FILE, sep='\t', usecols=[0, 1], names=['en', 'cn'],
                     encoding='utf-8').dropna()
    # 3.对原始语料做划分
    train_df, test_df = train_test_split(df, test_size=0.2)
    # 4.针对训练集分词，构建词表
    ChineseTokenizer.build_vocab(train_df['cn'].to_list(), MODEL_DIR / CN_VOCAB_FILE)
    EnglishTokenizer.build_vocab(train_df['en'].to_list(), MODEL_DIR / EN_VOCAB_FILE)
    # 5.创建分词器
    cn_tokenizer = ChineseTokenizer.from_vocab(MODEL_DIR / CN_VOCAB_FILE)
    en_tokenizer = EnglishTokenizer.from_vocab(MODEL_DIR / EN_VOCAB_FILE)
    # 6.构建数据集
    train_df['cn'] = train_df['cn'].apply(lambda text: cn_tokenizer.encode(text, mark=False))
    train_df['en'] = train_df['en'].apply(lambda text: en_tokenizer.encode(text, mark=True))
    test_df['cn'] = test_df['cn'].apply(lambda text: cn_tokenizer.encode(text, mark=False))
    test_df['en'] = test_df['en'].apply(lambda text: en_tokenizer.encode(text, mark=True))

    # 7.保存数据集到文件
    train_df.to_json(PROCESSED_DATA_DIR / TRAIN_DATA_FILE, orient='records', lines=True)
    test_df.to_json(PROCESSED_DATA_DIR / TEST_DATA_FILE, orient='records', lines=True)
    print("============开始数据预处理完成==============")


if __name__ == "__main__":
    preprocess()
