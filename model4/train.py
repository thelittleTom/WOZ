import torch
from  torch import  nn
from  torch import optim
from tqdm import tqdm
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader
from model import  WOZ
from WOZBERT.MyDataset import  MyDataset
from sklearn.metrics import f1_score
from datetime import datetime
current_time = datetime.now()
folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
BERT_PATH = '../Robert'
pad_size=80
epoch = 2000
batch_size = 248
lr = 1e-5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_path = './save/'+folder_name
labeltext2id={'景点': 0, '餐馆': 1, '酒店': 2, 'thank': 3, '出租': 4, '地铁': 5, 'bye': 6, '没有意图': 7, 'greet': 8}
lableid2text={0: '景点', 1: '餐馆', 2: '酒店', 3: 'thank', 4: '出租', 5: '地铁', 6: 'bye', 7: '没有意图', 8: 'greet'}
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(BERT_PATH)

InputDateset = MyDataset('../data/train.tsv',tokenizer,labeltext2id,pad_size)
DEvDateset = MyDataset('../data/val.tsv',tokenizer,labeltext2id,pad_size)
# 构建数据加载器
train_loader = DataLoader(InputDateset, batch_size=batch_size, shuffle=True)
dev_loader=DataLoader(DEvDateset, batch_size=batch_size)

model=WOZ(len(labeltext2id),BERT_PATH,device,pad_size).to(device)



l_params = list(map(id, model.bert.parameters()))
base_params = filter(lambda p: id(p) not in  l_params,
                     model.parameters())
optimizer = optim.AdamW([
        {"params":base_params},
        {"params":model.bert.parameters(),"lr":5e-5},],
        lr=1e-4, #默认参数
    )

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)


criterion =nn.CrossEntropyLoss(ignore_index=57)

def save_model(save_name):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_name))


best_dev_ma=0
best_dev_mi=0
for epoch_num in range(epoch):




    total_acc_train = 0
    total_loss_train = 0

    train_cnt=0
    for param_group in optimizer.param_groups:
        print(param_group['lr'])

    for text ,label in tqdm(train_loader):

        train_cnt+=1

        input_ids = text['input_ids'].squeeze(1).to(device)
        masks = text['attention_mask'].to(device)
        label = torch.as_tensor(label, dtype=torch.long, device=device)

        output = model(input_ids, masks)

        batch_loss = criterion(output, label.view(-1))
        batch_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        lr_scheduler.step()

        total_loss_train+=batch_loss.item()
    model.eval()
    total_acc_val = 0
    total_loss_val = 0
    # 不需要计算梯度
    cnt = 0
    alllabels=[]
    allPres=[]
    with torch.no_grad():
        # 循环获取数据集，并用训练好的模型进行验证
        for inputs, labels in dev_loader:
            cnt += 1
            input_ids = inputs['input_ids'].squeeze(1).to(device)  # torch.Size([32, 35])
            masks = inputs['attention_mask'].to(device)  # torch.Size([32, 1, 35])
            labels = labels.to(device)
            output = model(input_ids, masks)
            batch_loss = criterion(output, labels.view(-1))

            predicted_labels = torch.argmax(output, dim=1).cpu()  # 在最后一个维度上找到最大值的索引
            alllabels += labels.view(-1).cpu().tolist()
            allPres += predicted_labels.tolist()
            total_loss_val += batch_loss.item()
            # 计算准确率

        f1mascore = f1_score(alllabels, allPres, average='macro')
        f1miscore = f1_score(alllabels, allPres, average='micro')
        print(f'''Epochs: {epoch_num + 1} 
                  | Train Loss: {total_loss_train / train_cnt: .3f} 
                | Val f1micro: {f1miscore : .3f}
                | Val f1macrio: {f1mascore : .3f}
                | Val Loss: {total_loss_val / cnt : .3f}''')

        # 保存最优的模型
        if f1mascore > best_dev_ma :
            best_dev_ma = f1mascore
            stm = f"{f1miscore:.3f}_"
            stm+=f"{f1mascore:.3f}"
            save_model('best' + stm + '.pt')
        if f1miscore > best_dev_mi :
            best_dev_mi = f1miscore
            stm = f"{f1miscore:.3f}_"
            stm+=f"{f1mascore:.3f}"
            save_model('best' + stm + '.pt')




    model.train()
