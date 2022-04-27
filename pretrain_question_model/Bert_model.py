import torch
import numpy as np
import torch.nn as nn
from pretrain_question_model import Constants
from pretrain_question_model.bert_embedding import BERTEmbedding
from pretrain_question_model.model import BERT, CrossAttention, CrossBERT
from pretrain_question_model.task_model import Mask_skill, MaskDiff, Mask_question


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


def get_non_pad_mask(seq):
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_pad_mask(seq_k, seq_q):
    """For masking out the padding part of key sequence."""

    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x ls x lk
    return padding_mask


class BERTModel(nn.Module):
    """
      pretrain
    """

    def __init__(self, q_size, s_size, d_model, n_layers, d_k, d_v, n_head, d_inner, dropout):
        super().__init__()
        self.cross_attn = CrossAttention(temperature=np.power(d_model, 0.5), attn_dropout=dropout)
        # self.cross_trm = CrossBERT(d_model=d_model, n_layers=1, d_inner=d_inner, n_head=1, d_k=d_model, d_v=d_model, dropout=dropout)
        self.bert_question = BERT(d_model=d_model, n_layers=n_layers, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
        self.bert_skill = BERT(d_model=d_model, n_layers=n_layers, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout)
        self.bert_embedding = BERTEmbedding(q_vocab_size=q_size+5, s_vocab_size=s_size+5, embed_size=d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.task_predict_diff = MaskDiff(d_model)
        self.task_mask_question = Mask_question(d_model, q_size+1)
        self.task_mask_skill = Mask_skill(d_model, s_size+1)

    def forward(self, mask_question, mask_skill, question, total_skill):
        pad_mask_q = get_pad_mask(mask_question, mask_question)
        pad_mask_s = get_pad_mask(mask_skill, mask_skill)
        pad_dif_q = get_pad_mask(question, question)
        pad_dif_s = get_pad_mask(total_skill, total_skill)
        pad_dif_qs = get_pad_mask(total_skill, question)

        mask_q_emb, mask_s_emb = self.bert_embedding(mask_question, mask_skill)
        out_mask_q_emb = self.bert_question(mask_q_emb, mask=pad_mask_q)
        out_mask_s_emb = self.bert_skill(mask_s_emb, mask=pad_mask_s)

        q_emb, s_emb = self.bert_embedding(question, total_skill)

        out_q_emb = self.bert_question(q_emb, mask=pad_dif_q)
        out_s_emb = self.bert_skill(s_emb, mask=pad_dif_s)
        out_final_emb, cross_self_attn_list = self.cross_attn(out_q_emb, out_s_emb, out_s_emb, mask=pad_dif_qs)
        # out_final_emb, cross_self_attn_list = self.cross_trm(out_q_emb, out_s_emb, mask=pad_dif_qs)

        seq_logit_mask_ques = self.task_mask_question(out_mask_q_emb)
        seq_logit_mask_skill = self.task_mask_skill(out_mask_s_emb)
        seq_logit_diff_q = self.task_predict_diff(out_final_emb)

        return seq_logit_mask_ques, seq_logit_mask_skill, seq_logit_diff_q

    def get_embedding(self, question, skill):
        no_pad_mask = get_non_pad_mask(question)
        pad_mask_q = get_pad_mask(question, question)
        pad_mask_s = get_pad_mask(skill, skill)
        pad_mask_q_s = get_pad_mask(skill, question)

        dif_q_emb, dif_s_emb = self.bert_embedding(question, skill)

        dif_out_q_emb = self.bert_question(dif_q_emb, mask=pad_mask_q)
        dif_out_s_emb = self.bert_skill(dif_s_emb, mask=pad_mask_s)
        out_final_emb, cross_self_attn_list = self.cross_attn(dif_out_q_emb, dif_out_s_emb, dif_out_s_emb, mask=pad_mask_q_s)
        # out_final_emb, cross_self_attn_list = self.cross_trm(dif_out_q_emb, dif_out_s_emb, mask=pad_mask_q_s)

        out_final_emb *= no_pad_mask

        return out_final_emb, cross_self_attn_list







