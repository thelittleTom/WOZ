from torch import nn
from transformers import BertModel
import torch.nn.functional as F
import torch
import torch.nn as nn

# 定义一个 Transformer 编码器层

class WOZ(nn.Module):
    def __init__(self,num_labels,bertpath,device,pad_size=80):
        super(WOZ, self).__init__()
        filter_sizes = [3, 3]
        self.filter_num=512
        self.hidden_size=self.filter_num
        self.num_layers=2
        num_classes=num_labels
        self.bidirectional=False
        self.device=device
        self.dropout = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.3)
        self.bert = BertModel.from_pretrained(bertpath)
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=12, dim_feedforward=3072,dropout=0.5)
        # 定义一个 Transformer 编码器
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.padding_layer = nn.ZeroPad2d((0, 0, 1, 1))  # (left, right, top, bottom)


        self.conv1= nn.Conv2d(1, self.filter_num, (filter_sizes[0], 768))
        self.conv2= nn.Conv2d(1, self.filter_num, (filter_sizes[1], self.filter_num))

        self.lstm = nn.LSTM(input_size=self.filter_num, hidden_size=self.hidden_size, batch_first=True,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)
        self.fc = nn.Linear(self.hidden_size*pad_size, num_classes)


    def forward(self, input_id, mask):
        sequence_output, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        x= self.encoder(sequence_output)
        x = x.unsqueeze(1)
        x=self.padding_layer(x)
        x = F.relu(self.conv1(x)).squeeze(3)
        x=x.permute(0,2,1)
        x = x.unsqueeze(1)
        x=self.padding_layer(x)
        x = F.relu(self.conv2(x)).squeeze(3)
        x=x.permute(0,2,1)
        x = self.dropout(x)

        batch_size, seq_len,hid_size = x.shape

        h0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.randn(self.num_layers, batch_size, self.hidden_size).to(self.device)

        out, (_, _) = self.lstm(x, (h0, c0))
        # out=self.dropout2(out[:, -1, :])
        out = self.dropout2(out)
        out=out.reshape(out.size(0), -1)

        out = self.fc(out).squeeze(0)  # 因为有max_seq_len个时态，所以取最后一个时态即-1层

        return out