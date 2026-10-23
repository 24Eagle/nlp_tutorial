"""
#@time：2026/10/20  17:29
#@author：LiuYang
"""
import jieba
from config import *


class JiebaTokenizer():
    unk_token = UNK_TOKEN  # 类属性
    pad_token = PAD_TOKEN

    # 初始化
    def __init__(self, vocab_list):
        self.vovab_list = vocab_list
        self.vocab_size = len(vocab_list)  # 词表大小
        self.word2id = {word: id for id, word in enumerate(vocab_list)}
        self.id2word = {id: word for id, word in enumerate(vocab_list)}
        self.unk_id = self.word2id[self.unk_token]
        self.pad_id = self.word2id[self.pad_token]

    # 分词，静态方法
    @staticmethod
    def tokenize(text):
        return jieba.lcut(text)

    # 编码（将文本分词并id化）
    def encode(self, text,seq_len):
        # 分词
        tokens = self.tokenize(text)
        # 填充或截断到指定长度
        if len(tokens)>seq_len:
            tokens = tokens[:seq_len]
        elif len(tokens)<seq_len:
            tokens = tokens+[self.pad_token]*(seq_len-len(tokens))
        # id化
        ids = [self.word2id.get(token, self.unk_id) for token in tokens]
        return ids

    # 构建词表，并保存到文件
    @classmethod
    def build_vocab(cls, sentences, vocab_file_path):
        vocab_set = set()  # 利用集合做Token去重
        for sentence in sentences:
            vocab_set.update(jieba.lcut(sentence))
        # 转换成列表，（词表，id2word）,还要考虑未登录词和填充词
        vocab_list = [cls.pad_token,cls.unk_token] + list(vocab_set)
        print("词表大小：", len(vocab_list))
        # 5.保存词表到文件
        with open(vocab_file_path, "w", encoding="utf-8") as f:
            f.write('\n'.join(vocab_list))

    # 从文件加载词表，并创建分词器对象实例
    @classmethod
    def from_vocab(cls, vocab_file_path):
        # 1.获取词表
        with open(vocab_file_path, 'r', encoding='utf-8') as f:
            # 读取每一行
            vocab_list = [token.strip() for token in f.readlines()]
        # 构建分词对象
        tokenizer = cls(vocab_list)
        return tokenizer


if __name__ == "__main__":
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR / VOCAB_FILE)
    print("词表大小：", tokenizer.vocab_size)
    print("特殊符号unk：", tokenizer.unk_token)
    print("特殊符号PAD_ID：", tokenizer.pad_id)
    print(tokenizer.encode("自然语言处理",seq_len=SEQ_LEN))
