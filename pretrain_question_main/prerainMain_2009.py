import argparse
import time
import os
import torch
import torch.utils.data as Data
import torch.optim as optim
import torch.utils.data
from pretrain_question_model.Bert_model import BERTModel
from pretrain_question_main.pretrainDatasets import PretrainDataSet
from pretrain_question_main.pretraining import pretrain
from pretrain_question_main.optim_schedule import ScheduledOptim
from tensorboardX import SummaryWriter
from util import setup_seed

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
writer = SummaryWriter('runs/auc_2009')
setup_seed(42)


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch', type=int, default=400)
    parser.add_argument('-batch_size', type=int, default=32)
    parser.add_argument('-data_name', default='2009')

    parser.add_argument('-d_model', type=int, default=128)
    parser.add_argument('-d_inner_hid', type=int, default=512)
    parser.add_argument('-d_k', type=int, default=16)
    parser.add_argument('-d_v', type=int, default=16)

    parser.add_argument('-n_head', type=int, default=8)
    parser.add_argument('-n_layers', type=int, default=2)
    parser.add_argument('-n_warmup_steps', type=int, default=4000)
    parser.add_argument('-save_model', default='../savaMadel/')

    parser.add_argument('-dropout', type=float, default=0.2)
    parser.add_argument('-q_size', type=int, default=16891)
    parser.add_argument('-s_size', type=int, default=101)
    parser.add_argument('-step', type=int, default=5)
    parser.add_argument('-sq_len', type=int, default=256)
    parser.add_argument('-log', default=None)
    parser.add_argument('-no_cuda', action='store_true')

    opt = parser.parse_args()
    opt.cuda = not opt.no_cuda
    # ========= Loading Dataset =========
    print('*'*30, '数据加载中......', '*'*30)
    start = time.time()
    # =============data====================
    pretrain_data = torch.utils.data.DataLoader(PretrainDataSet(opt.data_name, opt.sq_len, opt.step, opt.q_size, opt.s_size), batch_size=opt.batch_size, shuffle=True)

    print('*' * 30, '耗时:', '{time:3.3f}s'.format(time=(time.time()-start)), '*' * 30)
    print('*' * 30, '参数打印中......', '*' * 30)
    print(opt)
    print('*' * 30, '训练模型加载中......', '*' * 30)
    device = torch.device('cuda' if opt.cuda else 'cpu')
    # =============pretain===============================
    print('pretrain............')
    pretrain_model = BERTModel(q_size=opt.q_size, s_size=opt.s_size, d_k=opt.d_k, d_v=opt.d_v, d_model=opt.d_model, d_inner=opt.d_inner_hid, n_layers=opt.n_layers, n_head=opt.n_head, dropout=opt.dropout).to(device)
    optimizer = ScheduledOptim(optim.Adam(pretrain_model.parameters(), betas=(0.9, 0.98), eps=1e-09), opt.d_model,
                               opt.n_warmup_steps)
    pretrain(pretrain_model, pretrain_data, optimizer, device, opt, opt.data_name, writer)


if __name__ == '__main__':
    main()
