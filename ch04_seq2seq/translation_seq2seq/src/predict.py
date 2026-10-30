"""
#@time：2026/10/17  13:26
#@author：LiuYang
"""
import torch
import jieba
from config import *
from model import ReviewAnalysisModel
from tokenizer import JiebaTokenizer


def predict_batch(model, inputs):
    """
    核心预测逻辑函数,返回一批数据的预测概率
    :param model:
    :param input:
    :param k:
    :return: topk的id
    """
    model.eval()
    # 前向传播
    with torch.no_grad():
        output = model(inputs)  # 形状(batch_size,)
    # 转化为预测概率
    batch_results = torch.sigmoid(output)
    return batch_results.tolist()  # 转换成列表返回


def predict(text, model, tokenizer, device):
    # 4.准备数据，文本处理
    # 4.1分词,id化
    ids = tokenizer.encode(text, SEQ_LEN)
    # 4.3转换为tensor，作为输入
    input = torch.tensor([ids], dtype=torch.long).to(device)
    # 5.核心预测逻辑
    # 5.1前向传播，得到预测概率
    result = predict_batch(model, input)
    return result[0]  # 只有唯一的一个数据


def run_predict():
    # 1.确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

    # 2.获取词表
    tokenizer = JiebaTokenizer.from_vocab(MODEL_DIR / VOCAB_FILE)
    print("词表加载成功")
    # 3.加载模型
    model = ReviewAnalysisModel(vocab_size=tokenizer.vocab_size,padding_idx=tokenizer.pad_id).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / BEST_MODEL))
    print("模型加载成功")

    # 6.程序运行流程
    print("欢迎使用文本情感分析模型！输入q或者quit退出......")
    input_history = ''
    while True:
        user_input = input(">")
        if user_input.strip() in ['q', 'quit']:
            print("欢迎下次再来")
            break
        elif user_input.strip() == '':
            print("请输入有效内容！")
            continue
        # 根据预测概率判断是正向还是负向评价
        result = predict(user_input,model,tokenizer,device)
        if result>0.5:
            print(f"正向评价（置信度：{result}）")
        else:
            print(f"负向评价（置信度：{1-result}）")


if __name__ == '__main__':
    # text = '我们公司'
    # top5_tokens = predict(text)
    # print(top5_tokens)
    run_predict()
