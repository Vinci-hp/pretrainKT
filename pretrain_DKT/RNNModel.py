import torch
import torch.nn as nn
from torch.autograd import Variable
from pretrain_DKT import Constants
from pretrain_question_main.pretraining import pretrain_embedding
# torch.backends.cudnn.enabled = False


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
        # self.LSTM = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=False)
        self.c_emb = nn.Embedding(5, input_dim, padding_idx=Constants.PAD_C)
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, q, full_s, s, c, pre_model=None):
        question_emb, _ = pretrain_embedding(model=pre_model, question=q, skill=full_s)
        correct_emb = self.c_emb(c)
        # x = qa_embed_data
        x = question_emb + correct_emb
        # x = torch.cat([question_emb, correct_emb], 2)
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda())
        # h0 = (Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()),
        #       Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).cuda()))
        out, hn = self.rnn(x, h0)
        res = self.sig(self.fc(out))
        return res
