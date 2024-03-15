from torch import nn
from transformers import BertModel
import torch.nn.functional as F
import torch
import torch.nn as nn

# 定义一个 Transformer 编码器层

class WOZ(nn.Module):
    def __init__(self,num_labels,bertpath,device,pad_size=None):
        super(WOZ, self).__init__()
        filter_sizes = [3, 4]
        self.filter_num=100
        self.hidden_size=self.filter_num
        self.num_layers=2
        num_classes=num_labels
        self.bidirectional=False
        self.device=device
        self.dropout = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.1)
        self.bert = BertModel.from_pretrained(bertpath)
        encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=4, dim_feedforward=3072,dropout=0.4)
        # 定义一个 Transformer 编码器
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.conv1= nn.Conv2d(1, self.filter_num, (filter_sizes[0], 768))
        self.conv2= nn.Conv2d(1, self.filter_num, (filter_sizes[1], self.filter_num))

        self.lstm = nn.LSTM(input_size=self.filter_num, hidden_size=self.hidden_size, batch_first=True,
                            num_layers=self.num_layers, bidirectional=self.bidirectional)
        if self.bidirectional:
            self.fc = nn.Linear(self.hidden_size * 2, num_classes)
        else:
            self.fc = nn.Linear(768, num_classes)


    def forward(self, input_id, mask):
        sequence_output, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)

        output = self.fc(pooled_output)  # 因为有max_seq_len个时态，所以取最后一个时态即-1层

        return output