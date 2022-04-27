
def readData(path):
    with open(path) as f:
        data = f.readlines()
    f.close()
    ex_ = []
    qx_ = []
    y_label_train = []
    m = int(len(data) / 4)
    for i in range(m):
        list_x = data[4 * i+1].replace('\n', '').split(',')
        list_x = list(map(lambda x: float(x), list_x[:-1]))
        # list_x = (np.array(list_x)+1).tolist()
        ex_.extend(list_x)
    print(ex_[len(ex_)-2:])
    print(len(ex_))
    li = set(ex_)
    print(len(li))
    print(max(li), min(li))

from random import randint

def getData(path):
    data = readData(path)
import torch
#  ----2009=16891----2017=3162----2012=53088------
# getData('data_question/assist2012_pid_train.csv')
# print(torch.version.cuda)