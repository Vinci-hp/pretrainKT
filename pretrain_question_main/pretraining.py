import time
from tqdm import tqdm
import torch
import os
import torch.nn as nn
import torch.utils.data
from pretrain_question_model import Constants
import torch.nn.functional as F


def multi_loss(pre_q, gold_q, pre_s, gold_s, pre_dif, gold_dif):
    mse = nn.MSELoss(reduction='mean')
    mask_q_loss = F.cross_entropy(pre_q, gold_q, ignore_index=Constants.PAD, reduction='mean')
    mask_s_loss = F.cross_entropy(pre_s, gold_s, ignore_index=Constants.PAD, reduction='mean')
    dif_loss = mse(pre_dif, gold_dif)
    # print('mask_q_loss=', mask_q_loss.item())
    # print('mask_s_loss=', mask_s_loss.item())
    # print('dif_loss=', dif_loss.item())
    loss = mask_q_loss + mask_s_loss + 30*dif_loss

    return loss


def get_mask(mask, d_model):
    mask = mask.data.cpu().numpy()
    mask = mask.reshape(-1, 1)
    mask = mask.repeat(d_model, axis=1)
    mask = torch.from_numpy(mask).cuda()

    return mask


def get_no_pad_value(pre, gold):
    d_model = pre.size(2)
    gold = gold.contiguous().view(-1, d_model)
    pre_ = pre.contiguous().view(-1, d_model)

    return pre_, gold


def get_question_or_skill_no_pad_value(pre, gold):
    d_model = pre.size(2)
    gold = gold.contiguous().view(-1)
    pre_ = pre.contiguous().view(-1, d_model)

    return pre_, gold


def get_diff_no_pad_value(pre, gold):
    gold = gold.contiguous().view(-1)
    pre_ = pre.contiguous().view(-1)
    no_mask = gold.ne(Constants.PAD_C)
    gold_dif = gold.masked_select(no_mask)
    pre_dif = pre_.masked_select(no_mask)
    return pre_dif, gold_dif


def pretrain_epoch(model, training_data, optimizer, device, opt):
    ''' Epoch operation in training phase'''

    model.train()
    total_loss = 0
    n_total = 0

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):

        batch_mask_q, batch_q_label, batch_mask_s, batch_s_label, batch_question, batch_total_skill, batch_diff_label = map(lambda x: x.to(device), batch)

        # forward
        optimizer.zero_grad()

        seq_logit_mask_question, seq_logit_mask_skill, seq_logit_diff_q = model(batch_mask_q, batch_mask_s, batch_question, batch_total_skill)

        # ====task-mask====
        pre_mask_q, gold_q = get_question_or_skill_no_pad_value(seq_logit_mask_question, batch_q_label)
        pre_mask_s, gold_s = get_question_or_skill_no_pad_value(seq_logit_mask_skill, batch_s_label)

        # ====task-difficult====
        pre_dif_q, gold_dif_q = get_diff_no_pad_value(seq_logit_diff_q, batch_diff_label)

        # backward
        loss = multi_loss(pre_mask_q, gold_q, pre_mask_s, gold_s, pre_dif_q, gold_dif_q)

        loss.backward()

        # update parameters
        optimizer.step_and_update_lr()
        # optimizer.step()
        total_loss += loss.item()

    return total_loss


def pretrain_embedding(model, question, skill):
    ''' get_pretrain_embedding'''
    model.eval()
    with torch.no_grad():
        # forward
        q_embed_data, cross_self_attn_list = model.get_embedding(question, skill)
        return q_embed_data, cross_self_attn_list


def pretrain(model, training_data, optimizer, device, opt, data_name, writer):
    min_los = []
    for epoch_i in range(opt.epoch):
        print('[ Epoch', epoch_i, ']')
        start = time.time()

        train_loss = pretrain_epoch(model, training_data, optimizer, device, opt)
        print('  - (Pre-train) loss: {loss: 3.5f}, time: {time:3.5f} s'.format(loss=train_loss, time=(time.time() - start)))

        min_los += [train_loss]
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'epoch': epoch_i}
        if epoch_i % 10 == 0:
            if train_loss <= min(min_los):
                model_name = opt.save_model + data_name + '/' + 'epoch_{epoch:3.4f}.chkpt'.format(epoch=epoch_i)
                torch.save(checkpoint, model_name)

                path = '../pretrain_log/log_' + data_name + '.txt'
                print('- [Info] The checkpoint file has been updated.')
                with open(path, 'a') as log_tf:
                    log_tf.truncate(0)
                    log_tf.write('epoch:{epoch: 5.0f}, test_auc:{test_auc: 5.3f}\n'.format(
                        epoch=epoch_i, test_auc=train_loss))

            writer.add_scalar('val_loss', train_loss, epoch_i)



