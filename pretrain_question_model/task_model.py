import torch
import torch.nn as nn


class Mask_question(nn.Module):

    def __init__(self, hidden, ques_sum):
        super().__init__()
        self.tgt = nn.Linear(hidden, ques_sum, bias=False)

    def forward(self, x):
        return self.tgt(x)


class Mask_skill(nn.Module):

    def __init__(self, hidden, skill_sum):
        super().__init__()
        self.tgt = nn.Linear(hidden, skill_sum, bias=False)

    def forward(self, x):
        return self.tgt(x)


class MaskR(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.tgt = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        return self.tgt(x)


class MaskDiff(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.tgt = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        return self.tgt(x)


class Predict_correct(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.tgt = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        return self.tgt(x)


class MatchR(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.tgt = nn.Linear(hidden, 2, bias=False)

    def forward(self, x):
        return self.tgt(x)


class MatchQ(nn.Module):

    def __init__(self, hidden):
        super().__init__()
        self.tgt = nn.Linear(hidden, 1, bias=False)

    def forward(self, x):
        return self.tgt(x)