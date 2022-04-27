import argparse
import time
import os
import torch
import numpy as np
import torch.utils.data as Data
import torch.optim as optim
import torch.utils.data
from pretrain_DKT.training import train
from pretrain_DKT.data_qr import getData, getDataSet
from pretrain_DKT.RNNModel import RNNModel
from pretrain_question_model.Bert_model import BERTModel
from util import setup_seed
setup_seed(10)
os.environ['CUDA_VISIBLE_DEVICES'] = '3'


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=100)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-d_model', type=int, default=128)
    parser.add_argument('-hidden', type=int, default=50)
    parser.add_argument('-layer', type=int, default=1)
    parser.add_argument('-out_dim', type=int, default=16891)
    parser.add_argument('-n_skill', type=int, default=101)
    parser.add_argument('-skill_size', type=int, default=101)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-log', default=None)
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    device = torch.device('cuda' if opt.cuda else 'cpu')
    #========= Loading Dataset =========#
    opt.max_token_seq_len = 50
    print('*'*30, '数据加载中......', '*'*30)
    start = time.time()
    # =====# 2009--110 --16891 2009update--138id--101name --16891   2015---100  2017--102--3162=======
    # =============data====================
    train_data, test_data, skill_full = getData('2009')
    training_data = torch.utils.data.DataLoader(getDataSet(train_data, skill_full), batch_size=opt.batch_size, shuffle=True)
    testing_data = torch.utils.data.DataLoader(getDataSet(test_data, skill_full), batch_size=opt.batch_size, shuffle=False)

    print('*' * 30, '耗时:', '{time:3.3f}s'.format(time=(time.time()-start)), '*' * 30)
    print('*' * 30, '参数打印中......', '*' * 30)
    print(opt)
    print('*' * 30, '训练模型加载中......', '*' * 30)
    # ==========load pretrain==========================
    pretrain = BERTModel(q_size=opt.out_dim, s_size=opt.n_skill, d_k=16,
                         d_v=16, d_model=128, d_inner=512, n_layers=2, n_head=8, dropout=0.2).to(device)
    checkpoint = torch.load('../pretrain_chkpt/2009-f.chkpt')
    pretrain.load_state_dict(checkpoint['model'])

    rnn = RNNModel(opt.d_model, opt.hidden, opt.layer, opt.out_dim, device).to(device)
    optimizer = optim.Adam(rnn.parameters(), lr=opt.lr)

    train(rnn, pretrain, training_data, testing_data, optimizer, device, opt, '2009')


if __name__ == '__main__':
    main()
