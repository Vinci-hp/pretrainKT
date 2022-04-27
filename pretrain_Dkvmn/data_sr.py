from torch.utils.data import Dataset
import numpy as np
from pretrain_Dkvmn import Constants

import torch

def readData(path, q_sum, sqlen):
    with open(path) as f:
        data = f.readlines()
    f.close()
    ex_ = []
    ex_s = []
    qx_ = []
    qx_s = []
    cor = []
    m = int(len(data)/4)
    for i in range(m):
        num = int(data[4*i].replace('\n', ''))
        list_s = data[4 * i + 2].replace('\n', '').split(',')
        list_s = list(map(lambda x: float(x), list_s))
        list_y = data[4 * i + 3].replace('\n', '').split(',')
        list_y = list(map(lambda x: float(x), list_y))
        if num <= sqlen:
            if num <= 2:
                continue
            x_ = np.array(list_s)
            y_ = np.array(list_y)
            ex = (x_ + y_ * q_sum).tolist()
            qx = (np.array(list_s)).tolist()
            c = (np.array(list_y)).tolist()
            ex_.append(ex)
            qx_.append(qx)
            cor.append(c)
        else:
            n_ = int(num / sqlen)
            for i in range(n_):
                x_1 = list_s[sqlen*i:sqlen*i+sqlen]
                y_1 = list_y[sqlen*i:sqlen*i+sqlen]
                x_ = np.array(x_1)
                y_ = np.array(y_1)
                ex = (x_ + y_ * q_sum).tolist()
                qx = (np.array(x_1)).tolist()
                c = (np.array(y_)).tolist()
                cor.append(c)
                ex_.append(ex)
                qx_.append(qx)
            if num % sqlen != 0:
                if num % sqlen > 2:
                    temx = list_s[-(num % sqlen):]
                    temy = list_y[-(num % sqlen):]
                    x_ = np.array(temx)
                    y_ = np.array(temy)
                    ex = (x_ + y_ * q_sum).tolist()
                    qx = (np.array(temx)).tolist()
                    c = (np.array(temy)).tolist()
                    cor.append(c)
                    ex_.append(ex)
                    qx_.append(qx)

    return ex_, qx_, cor


def getData(file_name, q_sum, sqlen):
    train_path = './data_question/'+file_name+'/assist' + file_name + '_pid_train.csv'
    test_path = './data_question/'+file_name+'/assist' + file_name + '_pid_test.csv'
    traindata = readData(train_path, q_sum, sqlen)
    testdata = readData(test_path, q_sum, sqlen)
    ex, qx, cor = traindata
    ex1, qx1, cor1 = testdata
    train_ex = np.array([e + [Constants.PAD] * (sqlen - len(e)) for e in ex], dtype=float)
    train_qx = np.array([e + [Constants.PAD] * (sqlen - len(e)) for e in qx], dtype=float)
    train_cor = np.array([e + [Constants.PAD_C] * (sqlen - len(e)) for e in cor], dtype=float)
    test_ex = np.array([e + [Constants.PAD] * (sqlen - len(e)) for e in ex1], dtype=float)
    test_qx = np.array([e + [Constants.PAD] * (sqlen - len(e)) for e in qx1], dtype=float)
    test_cor = np.array([e + [Constants.PAD_C] * (sqlen - len(e)) for e in cor1], dtype=float)

    return train_ex, train_qx, train_cor, test_ex, test_qx, test_cor

#
# train_ex, train_qx, train_cor, test_ex, test_qx, test_cor = getData('2009', 110, 50)
# print(train_ex.shape)
# print(train_ex[:2])
# print(train_qx[:2])
# print(train_cor[:2])