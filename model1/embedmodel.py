from torch import nn
from transformers import BertModel
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np

class Positional_Encoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out

# 定义一个 Transformer 编码器层

class WOZ(nn.Module):
    def __init__(self,num_labels,embedding_pretrained,device,pad_size=60):
        super(WOZ, self).__init__()
        filter_sizes = [3, 3]
        self.filter_num=300
        self.hidden_size=256
        self.num_layers=2
        num_classes=num_labels
        self.bidirectional=False
        self.device=device
        self.dropout = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        loaded_embeddings = torch.load(embedding_pretrained)

        self.embedding = nn.Embedding.from_pretrained( loaded_embeddings, freeze=False)
        self.postion_embedding = Positional_Encoding(300, pad_size,0.5, device)

        encoder_layer = nn.TransformerEncoderLayer(d_model=300, nhead=10, dim_feedforward=2048,dropout=0.5)
        # 定义一个 Transformer 编码器
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.padding_layer = nn.ZeroPad2d((0, 0, 1, 1))  # (left, right, top, bottom)



        self.conv1= nn.Conv2d(1, self.filter_num, (filter_sizes[0], 300))
        self.conv2= nn.Conv2d(1, self.filter_num, (filter_sizes[1], self.filter_num))

        self.lstm = nn.LSTM(input_size=self.filter_num, hidden_size=self.hidden_size, batch_first=True,
                            num_layers=self.num_layers, bidirectional=self.bidirectional,dropout=0.5)
        if self.bidirectional:
            self.fc = nn.Linear(self.hidden_size * 2, num_classes)
        else:
            self.fc = nn.Linear(self.hidden_size*pad_size, num_classes)
            # self.fc = nn.Linear(self.hidden_size*100, num_classes)



    def forward(self, input_id):
        x= self.embedding(input_id)
        x = self.postion_embedding(x)
        x= self.encoder(x)
        # out =x.view(x.size(0), -1)

        x = x.unsqueeze(1)
        x=self.padding_layer(x)
        x = F.relu(self.conv1(x)).squeeze(3)
        x=x.permute(0,2,1)
        x = x.unsqueeze(1)
        x=self.padding_layer(x)
        x = F.relu(self.conv2(x)).squeeze(3)
        x=x.permute(0,2,1)
        # x = self.dropout(x)
        # #
        # out = F.max_pool1d(x, x.size(2)).squeeze(2)

        out, (_, _) = self.lstm(x)


        # out=self.dropout2(out[:, -1, :])


        out=out.reshape(out.size(0), -1)
        out = self.fc(out).squeeze(0)
        # out = self.fc(out)
        return out

    # batch_size, seq_len,hid_size = x.shape
    # 初始化一个h0,也即c0，在RNN中一个Cell输出的ht和Ct是相同的，而LSTM的一个cell输出的ht和Ct是不同的
    # 维度[layers, batch, hidden_len]
    # if self.bidirectional:
    #     h0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
    #     c0 = torch.randn(self.num_layers * 2, batch_size, self.hidden_size).to(self.device)
    # else:
    #     h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
    #     c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
    # # out, (_, _) = self.lstm(x, (h0, c0))