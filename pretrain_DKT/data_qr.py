from torch.utils.data import Dataset
import numpy as np
import torch
from random import random, randint
from pretrain_DKT import Constants


def readData(path):
    with open(path) as f:
        data = f.readlines()
    f.close()
    ques_ = []
    skill_ = []
    corr_ = []
    qx = []
    gold_c = []
    m = int(len(data)/4)
    for i in range(m):
        num = int(data[4*i].replace('\n', ''))
        list_q = data[4 * i + 1].replace('\n', '').split(',')
        list_q = list(map(lambda x: float(x), list_q))
        list_s = data[4 * i + 2].replace('\n', '').split(',')
        list_s = list(map(lambda x: float(x), list_s))
        list_y = data[4 * i + 3].replace('\n', '').split(',')
        list_y = list(map(lambda x: float(x), list_y))
        if num <= 50:
            if num <= 2:
                continue
            ques_.append(np.array(list_q)[:-1].tolist())
            skill_.append(np.array(list_s)[:-1].tolist())
            corr_.append(np.array(list_y)[:-1].tolist())
            qx.append(np.array(list_q)[1:].tolist())
            gold_c.append(np.array(list_y)[1:].tolist())
        else:
            n_ = int(num / 50)
            for i in range(n_):
                x_1 = list_q[50*i:50*i+50]
                s_1 = list_s[50 * i:50 * i + 50]
                y_1 = list_y[50*i:50*i+50]
                ques_.append(np.array(x_1)[:-1].tolist())
                skill_.append(np.array(s_1)[:-1].tolist())
                corr_.append(np.array(y_1)[:-1].tolist())
                qx.append(np.array(x_1)[1:].tolist())
                gold_c.append(np.array(y_1)[1:].tolist())
            if num % 50 != 0:
                if num % 50 > 2:
                    temx = list_q[-(num % 50):]
                    tems = list_s[-(num % 50):]
                    temy = list_y[-(num % 50):]
                    ques_.append(np.array(temx)[:-1].tolist())
                    skill_.append(np.array(tems)[:-1].tolist())
                    corr_.append(np.array(temy)[:-1].tolist())
                    qx.append(np.array(temx)[1:].tolist())
                    gold_c.append(np.array(temy)[1:].tolist())

    return ques_, skill_, corr_, qx, gold_c


def get_skill(path):
    with open(path) as f:
        data = f.readlines()
    f.close()
    skill = []
    for i in data:
        skill = i.replace('\n', '').split(',')

    return skill


def getData(path):
    train_file = '../data_question/'+path+'/assist'+path+'_pid_train.csv'
    test_file = '../data_question/'+path+'/assist'+path+'_pid_test.csv'
    skill_path = '../data_question/' + path + '/assist' + path + '_skill.csv'

    skill_full = get_skill(skill_path)
    # train_file = 'builder'+path+'_train.csv'
    traindata = readData(train_file)
    testdata = readData(test_file)

    return traindata, testdata, skill_full


def getTensor(data, skill_full, max_len):
    ques, skill, corr, qx, gold_c = data
    b_ques = np.array([e + [Constants.PAD] * (max_len - len(e)) for e in ques], dtype=float)
    b_skill_full = np.array(skill_full, dtype=float)
    b_skill = np.array([e + [Constants.PAD] * (max_len - len(e)) for e in skill], dtype=float)
    b_corr = np.array([e + [Constants.PAD_C] * (max_len - len(e)) for e in corr], dtype=float)
    b_qx = np.array([e + [Constants.PAD] * (max_len - len(e)) for e in qx], dtype=float)
    b_gold_c = np.array([e + [Constants.PAD_C] * (max_len - len(e)) for e in gold_c], dtype=float)

    batch_Q = torch.from_numpy(b_ques).long()
    batch_full_s = torch.from_numpy(b_skill_full).long()
    batch_S = torch.from_numpy(b_skill).long()
    batch_c = torch.from_numpy(b_corr).long()
    batch_qx = torch.from_numpy(b_qx).long()
    batch_real_C = torch.from_numpy(b_gold_c).long()

    return batch_Q, batch_full_s, batch_S, batch_c, batch_qx, batch_real_C


class getDataSet(Dataset):
    def __init__(self, data, skill_full):
        self.max_len = 50
        self.dataset = getTensor(data, skill_full, max_len=self.max_len)
        self.batch_Q, self.batch_full_s, self.batch_S, self.batch_C, self.batch_qx, self.batch_gold = self.dataset
        assert (self.batch_Q.size(0) == self.batch_C.size(0) == self.batch_S.size(0))

        self.length = self.batch_Q.size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        q = self.batch_Q[idx]
        full_s = self.batch_full_s
        s = self.batch_S[idx]
        c = self.batch_C[idx]
        qx = self.batch_qx[idx]
        gold = self.batch_gold[idx]

        return q, full_s, s, c, qx, gold


# data, testdata, skill = getData('2009')
# training_data = torch.utils.data.DataLoader(getDataSet(data, skill), batch_size=128)
# for data in training_data:
#     q, full_s, s, c, qx, gold = data
#     print(q[:2])
#     print(full_s[:2])
#
#     break

# testing_data = torch.utils.data.DataLoader(testDataSet(testdata), batch_size=128)
# for data in testing_data:
#     Q, maskC, y = data
#     print(Q[:6])
#     print(maskC[:6])
#     print(y[:6])
#     break