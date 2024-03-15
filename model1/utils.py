# 生成初始权重

import numpy as np
vocab=[]

embedding_dim=300
with open("./sgns.weibo.char","r") as f:
    for line in f:
        words=line.strip().split(" ")
        if len(words[0])==1  :
            vocab.append(words[0])
        # elif len(words[0])==1 and words[0] in vocab:
        #     print(words[0])
addWord=[]
with open("../data/val.tsv","r", encoding="utf-8") as f:
    lines = f.readlines()
    i = 1
    while i < len(lines):
        words = lines[i].split(None)
        if words[1].startswith('"') and not words[1].endswith('"'):
            words[1] += lines[i + 1].split()[0]
            i += 1
        i+=1
        content=words[1]
        doc_tokens = [y for y in content]
        for word in doc_tokens:
            # 判断 word 是否在词汇表内
            if word in vocab:
               pass
            else:
                if word not in addWord:
                    addWord.append(word)
with open("../data/train.tsv","r", encoding="utf-8") as f:
    lines = f.readlines()
    i = 1
    while i < len(lines):
        words = lines[i].split(None)
        if words[1].startswith('"') and not words[1].endswith('"'):
            words[1] += lines[i + 1].split()[0]
            i += 1
        i+=1
        content=words[1]
        doc_tokens = [y for y in content]
        for word in doc_tokens:
            # 判断 word 是否在词汇表内
            if word in vocab:
               pass
            else:
                if word not in addWord:
                    addWord.append(word)
vocab=vocab+addWord
UNK, PAD = '<UNK>', '<PAD>'  # 未知字，padding符号
CLS='CLS'
vocab = [CLS] +vocab+[PAD, UNK]
vocab2id={}
for i,word in enumerate(vocab):
    vocab2id[word]=i
print(len(vocab2id))


import json

import torch

with open("./vocab1.json", "w") as f:
    json.dump(vocab2id, f)


# vocab2id = {word: idx for idx, word  in enumerate(vocab )}
# print(len(vocab2id))
pretrain_dir="./sgns.weibo.char"
embeddings = np.random.rand(len(vocab2id), embedding_dim)
f = open(pretrain_dir, "r", encoding='UTF-8')
for i, line in enumerate(f.readlines()):
    if i == 0:  # 若第一行是标题，则跳过
        continue
    lin = line.strip().split(" ")
    if lin[0] in vocab2id:
        idx = vocab2id[lin[0]]
        print(lin[0],idx)
        emb = [float(x) for x in lin[1:301]]
        embeddings[idx] = np.asarray(emb, dtype='float32')
f.close()
embeddings=torch.tensor(embeddings, dtype=torch.float32)
torch.save(embeddings, "./embedding_weights1.pt")