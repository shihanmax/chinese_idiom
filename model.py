import torch
import torch.nn as nn


class BinaryCls(nn.Module):
    
    def __init__(self, bert_model, bert_out_dim, hidden_dim, cls_dim=4):
        super(BinaryCls, self).__init__()
        self.bert_model = bert_model
        
        self.linear = nn.Linear(bert_out_dim, hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, cls_dim)
        self.tanh = nn.Tanh()
        
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
        self.bce = nn.BCELoss()
        self.nll = nn.NLLLoss()
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, **input):
        bert_out = self.bert_model(**input)  # bs, seq_len, hidden
        
        cls_repr = bert_out.pooler_output  # bs, hidden
        
        cls_repr = self.dropout(cls_repr)
        
        hid = self.tanh(self.linear(cls_repr))
        out_prob = self.log_softmax(self.out_linear(hid))  # bs, 4
        
        return out_prob
        
    def forward_loss(self, out_prob, label):
        loss = self.nll(out_prob, label)
        return loss
