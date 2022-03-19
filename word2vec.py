import collections
import os
import random
import zipfile
import numpy as np
import urllib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

#参数设置
EMBEDDING_DIM = 128 #词向量维度，植入神经网络的维度，一般是高维被降维
PRINT_EVERY = 100 #可视化频率
EPOCHES = 1000 #训练的轮数
BATCH_SIZE = 5 #每一批训练数据大小
N_SAMPLES = 3 #负样本大小
WINDOW_SIZE = 5 #周边词窗口大小
FREQ = 5 #词汇出现频数的阈值
DELETE_WORDS = False #是否删除部分高频词
VOCABULARY_SIZE = 50000

url='http://mattmahoney.net/dc/'#找其他skip-gram的数据集GitHub下载
def maybe_download(filename, expected_bytes):
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url+filename, filename)#后filename是保存的路径名
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:#验证下载的文件大小，普通文件以字节为单位的大小
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise  Exception('Failed to verify '+filename+'. Can you get to it with a browser?')
    return filename

filename=maybe_download('text8.zip', 31344016)

def read_data(filename):
    with zipfile.ZipFile(filename) as f:#打开一个 ZIP文件
        # 读取出来的每个单词是 bytes
        data=f.read(f.namelist()[0]).split()
        # import pdb
        # pdb.set_trace()
        # 把 bytes 转换为 str
        #data= [str(x, encoding = "utf8") for x in data] data文件里面是很多单词的序列
        data = list(map(lambda x: str(x, encoding = "utf8"), data))#用utf-8编码为字符串类型，并转换为list
    return data

words=read_data(filename)#单词列表
print('Data size', len(words))

# 取出频数前 50000 的单词 17005207个单词

counts_dict = dict((collections.Counter(words).most_common(VOCABULARY_SIZE-1)))#返回元组类型的列表，第一个是单词，第二个是频次
# 去掉频数小于 FREQ 的单词
# trimmed_words = [word for word in words if counts_dict[word] > FREQ]

# 计算 UNK 的频数 = 单词总数 - 前 50000 个单词的频数之和
counts_dict['UNK']=len(words)-np.sum(list(counts_dict.values()))#剩余单词频数,418391,counts_dict.values()单词频数列表

#建立词和索引的对应
idx_to_word = []
for word in counts_dict.keys():#所有单词列表，单词2单词个数
    idx_to_word.append(word)#下标是索引
word_to_idx = {word:i for i,word in enumerate(idx_to_word)}#单词2索引，根据counts_dict建立word_to_idx

#建立词和索引的对应
# idx_to_word = [word for word in counts_dict.keys()]
# word_to_idx = {word:i for i,word in enumerate(idx_to_word)}
# import pdb
# pdb.set_trace()
# 把前50000的单词列表转换为编号的列表
data=list()
for word in words:
    if word in word_to_idx:#
        index = word_to_idx[word]
    else:
        index=word_to_idx['UNK']#剩余所有的单词的频次都是总和的频次,word_to_idx['UNK']=49999,最大的索引
    data.append(index)

# 把单词列表转换为编号的列表
# data = [word_to_idx.get(word,word_to_idx["UNK"]) for word in words]

# 计算单词频次
total_count = len(data)#17005207
word_freqs = {w: c/total_count for w, c in counts_dict.items()}#计算counts_dict里面记录的单词的即前50000和剩余的单词总体
# 以一定概率去除出现频次高的词汇
if DELETE_WORDS:#默认不去除
    t = 1e-5
    # import pdb
    # pdb.set_trace()
    prob_drop = {w: 1-np.sqrt(t/word_freqs[w]) for w in data}#t/每个单词的频次
    data = [w for w in data if random.random()<(1-prob_drop[w])]#random.random()产生0，1间的随机数，如果随机数和t/频次的和小于1则选择该单词
else:
    data = data

#计算词频,按照原论文转换为3/4次方
word_counts = np.array([count for count in counts_dict.values()],dtype=np.float32)#频数
word_freqs = word_counts/np.sum(word_counts)#频率，仅类型转换
word_freqs = word_freqs ** (3./4.)#np数组总体计算，频率的3/4
word_freqs = word_freqs / np.sum(word_freqs)#频率的3/4 / 频率的3/4的总和


# DataLoader自动帮忙生成batch
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, data, word_freqs):
        super(WordEmbeddingDataset, self).__init__()#父类是抽象类
        self.data = torch.Tensor(data).long()  # 解码为词表中的索引
        self.word_freqs = torch.Tensor(word_freqs)  # 词频率 类型转换

    def __len__(self):
        # 共有多少个item
        return len(self.data)

    def __getitem__(self, idx):
        # 根据idx返回
        center_word = self.data[idx]  # 找到中心词
        pos_indices = list(range(idx - WINDOW_SIZE, idx)) + list(
            range(idx + 1, idx + WINDOW_SIZE + 1))  # 中心词前后各C个词作为正样本
        # pos_indices = list(filter(lambda i: i >= 0 and i < len(self.data), pos_indices))  # 过滤，如果索引超出范围，则丢弃
        pos_indices = [i % len(self.data) for i in pos_indices]
        pos_words = self.data[pos_indices]  # 周围单词
        # 根据 变换后的词频选择 K * 2 * C 个负样本，True 表示可重复采样
        neg_words = torch.multinomial(self.word_freqs, N_SAMPLES * pos_words.shape[0], True)

        return center_word, pos_words, neg_words


# 构造一个神经网络，输入词语，输出词向量
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        initrange = 0.5 / self.embed_size
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)#不使用稀疏矩阵
        # 模型输出nn.Embedding(30000, 100)
        self.out_embed.weight.data.uniform_(-initrange, initrange)  # 权重初始化的一种方法

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        # 模型输入nn.Embedding(30000, 100)
        self.in_embed.weight.data.uniform_(-initrange, initrange)
        # 权重初始化的一种方法

    def forward(self, input_labels, pos_labels, neg_labels):
        # input_labels:[batch_size]
        # pos_labels:[batch_size, windows_size*2]
        # neg_labels:[batch_size, windows_size * N_SAMPLES]

        input_embedding = self.in_embed(input_labels)  # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)  # [batch_size, windows_size * 2, embed_size]
        neg_embedding = self.out_embed(neg_labels)  # [batch_size, (windows_size * 2 * K), embed_size]

        # 向量乘法
        input_embedding = input_embedding.unsqueeze(2)  # [batch_size, embed_size, 1],新增一个维度用于向量乘法
        # input_embedding = input_embedding.view(BATCH_SIZE, EMBEDDING_DIM, 1)
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze(2)  # [batch_size, windows_size * 2] 只保留前两维
        neg_dot = torch.bmm(neg_embedding.neg(), input_embedding).squeeze(2)  # [batch_size, windows_size * 2 * K] 只保留前两维

        log_pos = F.logsigmoid(pos_dot).sum(1)  # 按照公式计算
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = -(log_pos + log_neg)  # [batch_size]

        return loss

    def input_embeddings(self):
        ##取出self.in_embed数据参数
        import pdb
        pdb.set_trace()
        return self.in_embed.weight.data.cpu().numpy()


# 构造  dataset 和 dataloader
dataset = WordEmbeddingDataset(data, word_freqs)#所有单词的表，特殊的频率
dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)#默认自动整理，不使用定制的sampler采样器

# 定义一个模型
model = EmbeddingModel(VOCABULARY_SIZE, EMBEDDING_DIM)
# import pdb
# pdb.set_trace()
# 定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)#随机梯度下降

for epoch in range(EPOCHES):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):

        input_labels = input_labels.long()  # 全部转为LongTensor
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()

        optimizer.zero_grad()  # 梯度归零，每次训练都进行
        # import pdb
        # pdb.set_trace()
        loss = model(input_labels, pos_labels, neg_labels).mean()#model(input_labels, pos_labels, neg_labels)返回tensor，平均的损失值
        loss.backward()
        optimizer.step()#每训练一次就更新一次参数

        if i % 100 == 0:
            print("epoch", epoch, "loss", loss.item())

    embedding_weights = model.input_embeddings()
    np.save("embedding-{}".format(EMBEDDING_DIM), embedding_weights)
    torch.save(model.state_dict(), "embedding-{}.th".format(EMBEDDING_DIM))



