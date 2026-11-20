"""
#@time：2026/10/17  13:26
#@author：LiuYang
"""
import torch
from tokenizer import ChineseTokenizer, EnglishTokenizer
from model import TranslationModel
import config

def predict_batch(input_tensor, model, en_tokenizer, device):
    """
    批量生成翻译结果。

    :param input_tensor: 中文输入张量 (batch_size, seq_len)。
    :param model: 翻译模型。
    :param en_tokenizer: 英文分词器。
    :param device: 设备。
    :return: 生成的英文索引列表。
    """
    model.eval()
    with torch.no_grad():
        src_pad_mask = (input_tensor == model.src_embedding.padding_idx)
        memory = model.encode(src=input_tensor, src_pad_mask=src_pad_mask)

        batch_size = input_tensor.shape[0]
        decoder_input = torch.full(
            size=(batch_size, 1),
            fill_value=en_tokenizer.sos_token_index,
            dtype=torch.long,
            device=device
        )

        generated = [[] for _ in range(batch_size)]
        finished = [False for _ in range(batch_size)]

        for step in range(1, config.SEQ_LEN):
            tgt_mask = model.transformer.generate_square_subsequent_mask(decoder_input.shape[1]).to(device)
            tgt_pad_mask = (decoder_input == en_tokenizer.pad_token_index)

            decoder_output = model.decode(decoder_input, memory, tgt_mask, tgt_pad_mask, src_pad_mask)

            predict_indexes = decoder_output[:, -1, :].argmax(dim=-1)

            for i in range(batch_size):
                if finished[i]:
                    continue
                if predict_indexes[i].item() == en_tokenizer.eos_token_index:
                    finished[i] = True
                    continue
                generated[i].append(predict_indexes[i].item())

            if all(finished):
                break

            # 自回归关键：必须把本步预测拼回 decoder_input，作为下一步输入。
            # 原代码把这句写在循环外，导致每一步都用只有 <sos> 的输入，永远只生成同一个词。
            # 已结束的样本后续用 <pad> 填充，避免继续产生无效 token。
            finished_tensor = torch.tensor(finished, dtype=torch.bool, device=device)
            next_input = torch.where(
                finished_tensor,
                torch.full_like(predict_indexes, en_tokenizer.pad_token_index),
                predict_indexes
            )
            decoder_input = torch.cat([decoder_input, next_input.unsqueeze(1)], dim=1)
    return generated

def predict(zh_sentence, model, zh_tokenizer, en_tokenizer, device):
    """
    翻译单句中文。

    :param zh_sentence: 中文句子。
    :param model: 翻译模型。
    :param zh_tokenizer: 中文分词器。
    :param en_tokenizer: 英文分词器。
    :param device: 设备。
    :return: 英文翻译句子。
    """
    input_ids = zh_tokenizer.encode(zh_sentence, seq_len=config.SEQ_LEN, add_sos_eos=False)
    input_tensor = torch.tensor([input_ids], device=device)
    generated = predict_batch(input_tensor, model, en_tokenizer, device)
    en_indexes = generated[0]
    en_sentence = en_tokenizer.decode(en_indexes)
    return en_sentence

def run_predict():
    """
    启动交互式翻译。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    zh_tokenizer = ChineseTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'zh_vocab.txt')
    en_tokenizer = EnglishTokenizer.from_vocab(config.PROCESSED_DATA_DIR / 'en_vocab.txt')

    model = TranslationModel(
        zh_vocab_size=zh_tokenizer.vocab_size,
        en_vocab_size=en_tokenizer.vocab_size,
        zh_padding_index=zh_tokenizer.pad_token_index,
        en_padding_index=en_tokenizer.pad_token_index
    ).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt', map_location=device))

    print('欢迎使用翻译系统，请输入中文句子：（输入 q 或 quit 退出）')
    while True:
        user_input = input('中文：')
        if user_input in ['q', 'quit']:
            print('谢谢使用，再见！')
            break
        if not user_input.strip():
            print('请输入内容')
            continue

        result = predict(user_input, model, zh_tokenizer, en_tokenizer, device)
        print(f'英文：{result}')

if __name__ == '__main__':
    run_predict()
