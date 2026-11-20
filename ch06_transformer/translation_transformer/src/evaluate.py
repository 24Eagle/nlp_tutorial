"""
#@time：2026/10/17  13:27
#@author：LiuYang
"""
import torch
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

import config
from tokenizer import ChineseTokenizer, EnglishTokenizer
from model import TranslationModel
from dataset import get_dataloader
from predict import predict_batch

def evaluate(dataloader, model, zh_tokenizer, en_tokenizer, device):
    all_references = []
    all_predictions = []
    # tgt 是英文编码，特殊符号索引必须取自英文词表（原代码误用了 zh_tokenizer）
    special_tokens = [en_tokenizer.pad_token_index, en_tokenizer.eos_token_index, en_tokenizer.sos_token_index]
    for src, tgt in tqdm(dataloader, desc="评估"):
        src = src.to(device)
        tgt = tgt.tolist()
        predict_indexes = predict_batch(src, model, en_tokenizer, device)
        all_predictions.extend([
            [index for index in pred if index not in special_tokens]
            for pred in predict_indexes
        ])

        for indexes in tgt:
            indexes = [index for index in indexes if index not in special_tokens]
            all_references.append([indexes])

    bleu = corpus_bleu(all_references, all_predictions)
    return bleu


def run_evaluate():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # tokenizer
    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'en_vocab.txt')

    # 模型
    model = TranslationModel(zh_vocab_size=zh_tokenizer.vocab_size,
        en_vocab_size = en_tokenizer.vocab_size,
        zh_padding_index = zh_tokenizer.pad_token_index,
        en_padding_index = en_tokenizer.pad_token_index).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt', map_location=device))
    # 数据
    dataloader = get_dataloader(train=False)

    bleu = evaluate(dataloader, model, zh_tokenizer, en_tokenizer, device)
    print(f'BLEU: {bleu:.2f}')

if __name__ == '__main__':
    run_evaluate()
