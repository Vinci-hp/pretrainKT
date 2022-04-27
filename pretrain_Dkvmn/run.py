import numpy as np
import math
import torch
import random
from torch import nn
import pretrain_Dkvmn.utils as utils
from sklearn import metrics


def train(epoch_num, model, pretrain, params, optimizer, q_data, qa_data, skill_full, train_skill, train_cor, device):
    N = int(math.floor(len(q_data) / params.batch_size))

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.train()

    # init_memory_value = np.random.normal(0.0, params.init_std, ())
    for idx in range(N):
        q_one_seq = q_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        qa_batch_seq = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        target = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        skill = train_skill[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        c_one = train_cor[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        target = (target - 1) / params.n_question
        target = np.floor(target)
        # embedding
        input_q = utils.varible(torch.LongTensor(q_one_seq), device)
        input_qa = utils.varible(torch.LongTensor(qa_batch_seq), device)
        input_skill_full = utils.varible(torch.LongTensor(skill_full), device)
        input_skill_full = input_skill_full.unsqueeze(0).expand(params.batch_size, -1)
        input_skill = utils.varible(torch.LongTensor(skill), device)
        input_c = utils.varible(torch.LongTensor(c_one), device)
        target = utils.varible(torch.FloatTensor(target), device)
        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)

        model.zero_grad()
        loss, filtered_pred, filtered_target = model.forward(input_q, input_qa, input_c, input_skill_full, input_skill, target_1d, pretrain)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), params.maxgradnorm)
        optimizer.step()
        epoch_loss += utils.to_scalar(loss)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())

        pred_list.append(right_pred)
        target_list.append(right_target)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)

    return epoch_loss/N, accuracy, auc


def test(model, pretrain, params, optimizer, q_data, qa_data, skill_full, test_skill, test_cor, device):
    N = int(math.floor(len(q_data) / params.batch_size))

    pred_list = []
    target_list = []
    epoch_loss = 0
    model.eval()

    # init_memory_value = np.random.normal(0.0, params.init_std, ())
    for idx in range(N):

        q_one_seq = q_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        qa_batch_seq = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        skill = test_skill[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        target = qa_data[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        one_c = test_cor[idx * params.batch_size:(idx + 1) * params.batch_size, :]
        target = (target - 1) / params.n_question
        target = np.floor(target)

        input_q = utils.varible(torch.LongTensor(q_one_seq), device)
        input_qa = utils.varible(torch.LongTensor(qa_batch_seq), device)
        input_skill_full = utils.varible(torch.LongTensor(skill_full), device)
        input_skill_full = input_skill_full.unsqueeze(0).expand(params.batch_size, -1)
        input_skill = utils.varible(torch.LongTensor(skill), device)
        input_c = utils.varible(torch.LongTensor(one_c), device)
        target = utils.varible(torch.FloatTensor(target), device)

        target_to_1d = torch.chunk(target, params.batch_size, 0)
        target_1d = torch.cat([target_to_1d[i] for i in range(params.batch_size)], 1)
        target_1d = target_1d.permute(1, 0)

        loss, filtered_pred, filtered_target = model.forward(input_q, input_qa, input_c, input_skill_full, input_skill, target_1d, pretrain)

        right_target = np.asarray(filtered_target.data.tolist())
        right_pred = np.asarray(filtered_pred.data.tolist())
        pred_list.append(right_pred)
        target_list.append(right_target)
        epoch_loss += utils.to_scalar(loss)

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    auc = metrics.roc_auc_score(all_target, all_pred)
    all_pred[all_pred >= 0.5] = 1.0
    all_pred[all_pred < 0.5] = 0.0
    accuracy = metrics.accuracy_score(all_target, all_pred)

    return epoch_loss/N, accuracy, auc









