import torch
from  torch import  nn
from  torch import optim
from tqdm import tqdm
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader
from embedmodel import  WOZ
from embedDataset import  MyDataset
from sklearn.metrics import f1_score
from datetime import datetime
current_time = datetime.now()
folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
BERT_PATH = './embedding_weights1.pt'
modelPath="./best0.945_0.932.pt"

batch_size = 128

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

labeltext2id={'景点': 0, '餐馆': 1, '酒店': 2, 'thank': 3, '出租': 4, '地铁': 5, 'bye': 6, '没有意图': 7, 'greet': 8}
lableid2text={0: '景点', 1: '餐馆', 2: '酒店', 3: 'thank', 4: '出租', 5: '地铁', 6: 'bye', 7: '没有意图', 8: 'greet'}
vocab_path="./vocab1.json"
InputDateset = MyDataset('../data/test.tsv', labeltext2id, vocab_path,pad_size=60)


# 构建数据加载器
test_loader = DataLoader(InputDateset, batch_size=batch_size)


model=WOZ(len(labeltext2id),BERT_PATH,device,pad_size=60).to(device)
model.load_state_dict(torch.load(modelPath) )


alllabels=[]
allPres=[]
with torch.no_grad():
    # 循环获取数据集，并用训练好的模型进行验证
    i=0
    for inputs, labels in test_loader:
        input_ids = torch.stack(inputs).to(device)
        input_ids = input_ids.t()
        labels = torch.as_tensor(labels, dtype=torch.long, device=device)
        if i==30:
            continue
            print(i)
        i+=1
        output = model(input_ids)


        predicted_labels = torch.argmax(output, dim=1).cpu()  # 在最后一个维度上找到最大值的索引
        alllabels += labels.view(-1).cpu().tolist()
        allPres += predicted_labels.tolist()

        # 计算准确率

    f1mascore = f1_score(alllabels, allPres, average='macro')
    f1miscore = f1_score(alllabels, allPres, average='micro')

    for i,(pre,label) in enumerate(zip(allPres,alllabels)):
        if pre!=label:
            print(i+2,lableid2text[label],lableid2text[pre])
    print(f'''
              | test  f1miscore: {f1miscore: .3f} 
            | test f1masocre: {f1mascore : .3f}''')