"""
#@time：2026/10/17  13:26
#@author：LiuYang
"""
import torch
import jieba
from config import *
from model import InputMethodRNNModel


def predict_topk(model, input, k=5):
    """
    核心预测逻辑函数
    :param model:
    :param input:
    :param k:
    :return: topk的id
    """
    model.eval()
    # 前向传播
    with torch.no_grad():
        output = model(input)  # 形状(batch_size,vocab_size)
    # 取输出的topk
    # >> > torch.topk(x, 3)返回x中最大的3个数
    # values就是最大的那三个数，然后indices就是每个数对应的索引
    # torch.return_types.topk(values=tensor([5., 4., 3.]), indices=tensor([4, 3, 2]))
    # topk的id列表
    top_indices = torch.topk(output, k).indices
    return top_indices.tolist()  # 转换成列表返回


def predict(text, model, id2word, word2id, k, device):
    # 4.准备数据，文本处理
    # 4.1分词
    tokens = jieba.lcut(text)
    # 4.2id化
    ids = [word2id.get(token, word2id.get(UNK_TOKEN)) for token in tokens]
    # 4.3转换为tensor，作为输入
    input = torch.tensor([ids], dtype=torch.long).to(device)
    # 5.核心预测逻辑
    # 5.1前向传播，得到topk的id列表
    top_indices_list = predict_topk(model, input, k=k)
    # 5.2将id列表转换成token列表
    top_tokens = [id2word[id] for id in top_indices_list[0]]
    return top_tokens


def run_predict():
    # 1.确定设备
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    # 2.获取词表
    with open(MODEL_DIR / VOCAB_FILE, 'r', encoding='utf-8') as f:
        # 读取每一行
        vocab_list = [token.strip() for token in f.readlines()]
    id2word = {id: word for id, word in enumerate(vocab_list)}
    word2id = {word: id for id, word in enumerate(vocab_list)}
    print("词表加载成功")
    # 3.加载模型
    model = InputMethodRNNModel(vocab_size=len(vocab_list)).to(device)
    model.load_state_dict(torch.load(MODEL_DIR / BEST_MODEL))
    print("模型加载成功")

    # 6.程序运行流程
    print("欢迎使用智能输入法模型！输入q或者quit退出......")
    input_history = ''
    while True:
        user_input = input(">")
        if user_input.strip() in ['q', 'quit']:
            print("欢迎下次再来")
            break
        elif user_input.strip() == '':
            print("请输入有效内容！")
            continue
        input_history += user_input
        # 预测tokens
        top_indices = predict(input_history, model, id2word, word2id, 5, device)
        print("预测结果：", top_indices)


if __name__ == '__main__':
    # text = '我们公司'
    # top5_tokens = predict(text)
    # print(top5_tokens)
    run_predict()
