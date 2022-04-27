''' Define the bert model '''
import torch.nn as nn
import torch
from pretrain_question_model.transformer import CrossTransformerBlock, TransformerBlock
from pretrain_question_model import Constants
import numpy as np


def get_pad_mask(seq):
    ''' For masking out the padding part of key sequence. '''
    len_q = seq.size(1)
    padding_mask = seq.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


class CrossAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2) #d

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class CrossBERT(nn.Module):
    """
    CrossBERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, d_model, n_layers, d_k, d_v, n_head, d_inner, dropout):
        super().__init__()
        self.cross_transformer = nn.ModuleList([CrossTransformerBlock(d_k, d_v, d_model, d_inner, n_head, dropout=dropout) for _ in range(n_layers)])

    def forward(self, input_1, input_2, mask=None, return_attn=True):
        cross_self_attn_list = []
        if mask is not None:
            cross_output = input_1
            for cro_transformer in self.cross_transformer:
                cross_output, cross_self_attn = cro_transformer(cross_output, input_2, mask)

                if return_attn:
                    cross_self_attn_list += [cross_self_attn]

            if return_attn:
                return cross_output, cross_self_attn_list
            return cross_output


class BERT(nn.Module):
    """
    CrossBERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, d_model, n_layers, d_k, d_v, n_head, d_inner, dropout):
        super().__init__()
        self.transformers = nn.ModuleList([TransformerBlock(d_k, d_v, d_model, d_inner, n_head, dropout=dropout) for _ in range(n_layers)])

    def forward(self, input_, mask=None):
        if mask is not None:
            output = input_
            for transformer in self.transformers:
                output = transformer(output, mask)
            return output