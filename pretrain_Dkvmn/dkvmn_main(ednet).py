import torch
import argparse
from pretrain_Dkvmn.model import MODEL
from pretrain_Dkvmn.run import train, test
import torch.optim as optim
import time
from pretrain_Dkvmn.data import getData
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from pretrain_question_model.Bert_model import BERTModel
import numpy as np
import os
from utils import setup_seed
setup_seed(20)
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_iter', type=int, default=50, help='number of iterations')
    parser.add_argument('--decay_epoch', type=int, default=20, help='number of iterations')
    parser.add_argument('--test', type=bool, default=False, help='enable testing')
    parser.add_argument('--train_test', type=bool, default=True, help='enable testing')
    parser.add_argument('--show', type=bool, default=True, help='print progress')
    parser.add_argument('--init_std', type=float, default=0.1, help='weight initialization std')
    parser.add_argument('--init_lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.75, help='learning rate decay')
    parser.add_argument('--final_lr', type=float, default=1E-5,
                        help='learning rate will not decrease after hitting this threshold')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum rate')
    parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
    parser.add_argument('--final_fc_dim', type=float, default=128, help='hidden state dim for final fc layer')

    dataset = 'assist_ednet'

    if dataset == 'assist2009_updated':
        parser.add_argument('--batch_size', type=int, default=128, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=128, help='question embedding dimensions')
        parser.add_argument('--qa_embed_dim', type=int, default=128, help='answer and question embedding dimensions')
        parser.add_argument('--memory_size', type=int, default=20, help='memory size')
        parser.add_argument('--n_question', type=int, default=16891, help='the number of unique questions in the dataset')
        parser.add_argument('--n_skill', type=int, default=101, help='the number of unique questions in the dataset')
        parser.add_argument('--skill_size', type=int, default=101, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=50, help='the allowed maximum length of a sequence')
        parser.add_argument('--save', type=str, default='assist2009_updated', help='path to save model')

    elif dataset == 'assist2012':
        parser.add_argument('--batch_size', type=int, default=128, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=128, help='question embedding dimensions')
        parser.add_argument('--qa_embed_dim', type=int, default=128, help='answer and question embedding dimensions')
        parser.add_argument('--memory_size', type=int, default=20, help='memory size')
        parser.add_argument('--n_question', type=int, default=50983, help='the number of unique questions in the dataset')
        parser.add_argument('--n_skill', type=int, default=198, help='the number of unique questions in the dataset')
        parser.add_argument('--skill_size', type=int, default=198, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=50, help='the allowed maximum length of a sequence')
        parser.add_argument('--load', type=str, default='assist2012_pid', help='model file to load')
        parser.add_argument('--save', type=str, default='assist2012_pid', help='path to save model')

    elif dataset == 'assist_ednet':
        parser.add_argument('--batch_size', type=int, default=128, help='the batch size')
        parser.add_argument('--q_embed_dim', type=int, default=128, help='question embedding dimensions')
        parser.add_argument('--qa_embed_dim', type=int, default=128, help='answer and question embedding dimensions')
        parser.add_argument('--memory_size', type=int, default=20, help='memory size')
        parser.add_argument('--n_question', type=int, default=12368, help='the number of unique questions in the dataset')
        parser.add_argument('--n_skill', type=int, default=1901, help='the number of unique questions in the dataset')
        parser.add_argument('--skill_size', type=int, default=188, help='the number of unique questions in the dataset')
        parser.add_argument('--seqlen', type=int, default=50, help='the allowed maximum length of a sequence')
        parser.add_argument('--load', type=str, default='assist2012_pid', help='model file to load')
        parser.add_argument('--save', type=str, default='assist2012_pid', help='path to save model')

    device = torch.device('cuda')
    params = parser.parse_args()
    params.lr = params.init_lr
    params.memory_key_state_dim = params.q_embed_dim
    params.memory_value_state_dim = params.qa_embed_dim

    print(params)
    # ========xiugai===========
    train_qa_data, train_q_data, skill_full, train_skill, train_cor, valid_qa_data, valid_q_data, test_skill, test_cor = getData('ednet', params.n_question, params.seqlen)
    print(params.n_question)

    # ========xiugai===========
    params.memory_key_state_dim = params.q_embed_dim  # 50
    params.memory_value_state_dim = params.qa_embed_dim # 200

    model = MODEL(n_question=params.n_question,
                  batch_size=params.batch_size,
                  q_embed_dim=params.q_embed_dim,
                  qa_embed_dim=params.qa_embed_dim,
                  memory_size=params.memory_size,
                  memory_key_state_dim=params.memory_key_state_dim,
                  memory_value_state_dim=params.memory_value_state_dim,
                  final_fc_dim=params.final_fc_dim)

    # ==========load pretrain==========================
    pretrain = BERTModel(q_size=params.n_question, s_size=params.n_skill, d_k=16,
                         d_v=16, d_model=128, d_inner=512, n_layers=2, n_head=8, dropout=0.2).to(device)
    checkpoint = torch.load('../pretrain_chkpt/ednet-50.chkpt')
    pretrain.load_state_dict(checkpoint['model'])

    # pretrain_qa = BERT4Model(q_size=params.n_question, s_size=params.n_skill, d_k=32, d_v=32, d_model=128, d_inner=512,
    #                          n_layers=1, n_head=4, dropout=0.1).to(device)
    # checkpoint = torch.load('../pretrain_chkpt/ednet_qa.chkpt')
    # pretrain_qa.load_state_dict(checkpoint['model'])

    # init_ q,qa embedding
    model.init_embeddings()
    model.init_params()
    optimizer = optim.Adam(params=model.parameters(), lr=params.lr, betas=(0.9, 0.9))

    # if params.gpu >= 0:
    # print('device: ' + str(device))
    #     torch.cuda.set_device(params.gpu)
    model.to(device)
    train_ = []
    test_ = []
    writer = SummaryWriter('runs/auc_ednet')
    for idx in range(params.max_iter):

        start = time.time()
        train_loss, train_accuracy, train_auc = train(idx, model, pretrain, params, optimizer, train_q_data, train_qa_data, skill_full, train_skill, train_cor, device)
        print('Epoch {idx:3d}/{total:3d} - (Train) loss: {loss: 3.5f}, train_auc: {train_auc: 5.3f}, time: {time:3.5f} s'.format(idx=idx+1, total=params.max_iter, loss=train_loss, train_auc=train_auc*100, time=(time.time() - start)))
        train_ += [train_auc]
        start = time.time()
        test_loss, test_accuracy, test_auc = test(model, pretrain, params, optimizer, valid_q_data, valid_qa_data, skill_full, test_skill, test_cor, device)
        print('              - (Test) loss: {loss: 3.5f}, test_auc: {test_auc: 5.3f}, time: {time:3.5f} s'.format(loss=test_loss, test_auc=test_auc*100, time=(time.time() - start)))

        test_ += [test_auc]

        if test_auc >= max(test_):
            print('-[Info] The best_auc:{auc:3.5f}'.format(auc=test_auc*100))
            with open('./log/log_ednet.txt', 'a') as log_tf:
                log_tf.truncate(0)

                log_tf.write('epoch:{epoch: 5.0f}, test_auc:{test_auc: 5.5f}, time: {time:5.3f} s\n'.format(
                    epoch=idx, test_auc=test_auc, time=(time.time() - start)))
        writer.add_scalars('', {'train_auc': train_auc, 'test_auc': test_auc}, idx)


if __name__ == "__main__":
    main()
