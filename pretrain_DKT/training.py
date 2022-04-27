import time
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data
from Metrics import auc
from pretrain_DKT import Constants
from tensorboardX import SummaryWriter


def lossFunc(pre, gold):
    BCE = nn.BCELoss(reduction='sum')
    loss = BCE(pre, gold)

    return loss


def get_mask(mask, d_model):
    mask = mask.data.cpu().numpy()
    mask = mask.reshape(-1, 1)
    mask = mask.repeat(d_model, axis=1)
    mask = torch.from_numpy(mask).cuda()

    return mask


def get_no_pad_real(pre_emb, gold_index, gold_C):
    d_model = pre_emb.size(-1)
    index = gold_index
    gold = gold_C
    pre_ = pre_emb.contiguous().view(-1, d_model)
    no_mask = gold_index.ne(Constants.PAD)
    index_ = index.masked_select(no_mask)
    index_ = index_ - 1
    gold_ = gold.masked_select(no_mask)
    index_ = index_.unsqueeze(0)
    pre_mask = get_mask(no_mask, d_model)
    index_ = index_.t()
    pre_a = pre_.masked_select(pre_mask).reshape(-1, d_model)

    pre_ = pre_a.gather(index=index_, dim=1).t().squeeze(0)

    return pre_, gold_


def train_epoch(model, pretrain, training_data, optimizer, device, opt):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0
    n_total = 0
    y_pred = torch.zeros([1], dtype=torch.float).to(device)
    y_gold = torch.zeros([1], dtype=torch.float).to(device)
    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):

        q, full_s, s, c, qx, gold = map(lambda x: x.to(device), batch)

        # forward
        optimizer.zero_grad()

        pre = model(q, full_s, s, c, pre_model=pretrain)

        # ====task1-Predict_correct====
        task1_pre, task1_gold = get_no_pad_real(pre, qx, gold)
        task1_gold = task1_gold.float()

        # backward
        # loss = cal_performance(task1_pre, task1_gold, task2_pre, task2_gold)
        loss = lossFunc(task1_pre, task1_gold)
        loss.backward()

        y_pred = torch.cat([y_pred, task1_pre], dim=0)
        y_gold = torch.cat([y_gold, task1_gold], dim=0)
        # update parameters
        optimizer.step()

        total_loss += loss.item()

        gold_num = task1_gold.size(0)
        n_total += gold_num
    mean_loss = total_loss / n_total
    ac_pred = y_pred[1:].data.cpu().numpy()
    ac_gold = y_gold[1:].data.cpu().numpy()
    auc_ = auc(ac_gold, ac_pred)
    return mean_loss, auc_


def test_epoch(model, pretrain, testing_data, device):
    ''' Epoch operation in training phase'''

    model.eval()

    total_loss = 0
    n_total = 0
    y_pred = torch.zeros([1], dtype=torch.float).to(device)
    y_gold = torch.zeros([1], dtype=torch.float).to(device)
    with torch.no_grad():
        for batch in tqdm(testing_data, mininterval=2, desc='  - (Training)   ', leave=False):
            q, full_s, s, c, qx, gold = map(lambda x: x.to(device), batch)
            # forward
            pre = model(q, full_s, s, c, pre_model=pretrain)

            # ====task1-Predict_correct====
            task1_pre, task1_gold = get_no_pad_real(pre, qx, gold)
            task1_gold = task1_gold.float()

            # backward
            loss = lossFunc(task1_pre, task1_gold)

            y_pred = torch.cat([y_pred, task1_pre], dim=0)
            y_gold = torch.cat([y_gold, task1_gold], dim=0)

            total_loss += loss.item()

            gold_num = task1_gold.size(0)
            n_total += gold_num
    mean_loss = total_loss / n_total
    ac_pred = y_pred[1:].data.cpu().numpy()
    ac_gold = y_gold[1:].data.cpu().numpy()
    auc_ = auc(ac_gold, ac_pred)
    return mean_loss, auc_


def train(model, pretrain, training_data, testing_data, optimizer, device, opt, data_name):
    '''Start training'''
    writer = None
    if data_name == '2009':
        writer = SummaryWriter('runs/auc_2009')
    elif data_name == '2017':
        writer = SummaryWriter('runs/auc_2017')
    elif data_name == '2012':
        writer = SummaryWriter('runs/auc_2012')
    elif data_name == 'ednet':
        writer = SummaryWriter('runs/auc_ednet')
    train = []
    test = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')
        start = time.time()
        train_loss, train_auc = train_epoch(model, pretrain, training_data, optimizer, device, opt)
        print('  - (Test) loss: {loss: 3.5f}, train_auc: {train_auc: 3.5f}, time: {time:3.5f} s'.format(loss=train_loss, train_auc=train_auc*100, time=(time.time() - start)))
        train += [train_auc]
        test_loss, test_auc = test_epoch(model, pretrain, testing_data, device)
        print('  - (Test) loss: {loss: 3.5f}, test_auc: {test_auc: 3.5f}, time: {time:3.5f} s'.format(loss=test_loss, test_auc=test_auc*100, time=(time.time() - start)))
        test += [test_auc]

        if test_auc >= max(test):
            print('- [Info] The test-auc has been updated.')
            with open('./dkt_log/log_'+data_name+'.txt', 'a') as log_tf:
                log_tf.truncate(0)
                log_tf.write('epoch:{epoch: 5.0f}, test_auc:{test_auc: 5.5f}, time: {time:5.3f} s\n'.format(
                    epoch=epoch_i, test_auc=test_auc, time=(time.time() - start)))
        writer.add_scalars('', {'train_auc': train_auc, 'test_auc': test_auc}, epoch_i)


