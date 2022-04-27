from torch.utils.data import Dataset
from DatasetMethods import DatasetMethods
import numpy as np
import torch
from random import random, randint
from pretrain_question_model import Constants


class PretrainDataSet(Dataset):
    def __init__(self, data_name, sq_len, step, question_size, skill_size):
        self.max_len = sq_len
        self.getPretrainDataSet = GetPretrainDataSet(data_name, sq_len, step, question_size, skill_size, self.max_len)
        self.batch_mask_questions, self.batch_question_label, self.batch_mask_total_skills, self.batch_total_skill_label,\
        self.batch_question, self.batch_total_skills, self.difficult_labels = self.getPretrainDataSet.getAllTensorData()

        assert (self.batch_mask_questions.size(0) == self.batch_mask_total_skills.size(0) == self.batch_question.size(0) == self.batch_total_skills.size(0))
        self.length = self.batch_mask_questions.size(0)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        batch_mask_q = self.batch_mask_questions[idx]
        batch_q_label = self.batch_question_label[idx]
        batch_mask_s = self.batch_mask_total_skills[idx]
        batch_s_label = self.batch_total_skill_label[idx]
        batch_question = self.batch_question[idx]
        batch_total_skill = self.batch_total_skills[idx]
        batch_diff_label = self.difficult_labels[idx]

        return batch_mask_q, batch_q_label, batch_mask_s, batch_s_label, batch_question, batch_total_skill, batch_diff_label


class GetPretrainDataSet:
    def __init__(self, data_name, sq_len, step, question_size, skill_size, max_len):
        self.max_len = max_len
        self.question_size = question_size
        self.skill_size = skill_size
        self.dataset_methods = DatasetMethods(data_name, sq_len, step)

    def getAllTensorData(self):
        train_data, shuffle_total_skills, single_total_skills, difficult_dict = self.dataset_methods. \
            getTrainDataAndSkillDataAndDifficultDict()
        batch_mask_q, batch_q_label = self.getTrainDataTensor(train_data, self.question_size, self.max_len)
        batch_mask_s, batch_s_label = self.getTotalSkillDataTensor(shuffle_total_skills, self.skill_size)
        batch_question, batch_total_skill, batch_difficult_labels = self.getDifficultTrainDataTensor(train_data, single_total_skills, difficult_dict, self.max_len)

        return batch_mask_q, batch_q_label, batch_mask_s, batch_s_label, batch_question, batch_total_skill, batch_difficult_labels

    def getTrainDataTensor(self, data, question_size, max_len):
        # =======mask===========
        mask_questions, question_label = self.maskedTrainData(data, question_size)

        b_mask_questions = np.array([e + [Constants.PAD] * (max_len - len(e)) for e in mask_questions], dtype=float)
        b_question_label = np.array([e + [Constants.PAD] * (max_len - len(e)) for e in question_label], dtype=float)

        batch_mask_questions = torch.from_numpy(b_mask_questions).long()
        batch_question_label = torch.from_numpy(b_question_label).long()

        return batch_mask_questions, batch_question_label

    def maskedTrainData(self, data, question_size):
        mask_questions = []
        question_labels = []
        quest, skill = data
        data_zip = zip(quest, skill)
        for data_ in data_zip:
            mask_q, q_label = GetPretrainDataSet.maskedSingleData(data_, question_size, 'question')
            mask_questions.append(mask_q)
            question_labels.append(q_label)

        return mask_questions, question_labels

    def getTotalSkillDataTensor(self, data, skill_size):
        mask_total_skills = []
        total_skill_labels = []
        for data_ in data:
            mask_s, s_label = GetPretrainDataSet.maskedSingleData(data_, skill_size, 'skill')
            mask_total_skills.append(mask_s)
            total_skill_labels.append(s_label)

        b_mask_total_skills = np.array(mask_total_skills, dtype=float)
        b_total_skill_labels = np.array(total_skill_labels, dtype=float)

        batch_mask_total_skills = torch.from_numpy(b_mask_total_skills).long()
        batch_total_skill_labels = torch.from_numpy(b_total_skill_labels).long()

        return batch_mask_total_skills, batch_total_skill_labels

    def getDifficultTrainDataTensor(self, data, single_total_skills, difficult_dict, max_len):
        questions, total_skills, difficult_labels = self.getTrainDifficultData(data, single_total_skills, difficult_dict)
        b_questions = np.array([e + [Constants.PAD] * (max_len - len(e)) for e in questions], dtype=float)
        b_total_skills = np.array(total_skills, dtype=float)
        b_difficult_labels = np.array([e + [Constants.PAD_C] * (max_len - len(e)) for e in difficult_labels], dtype=float)

        batch_questions = torch.from_numpy(b_questions).long()
        batch_total_skills = torch.from_numpy(b_total_skills).long()
        batch_difficult_labels = torch.from_numpy(b_difficult_labels).float()

        return batch_questions, batch_total_skills, batch_difficult_labels

    def getTrainDifficultData(self, data, single_total_skills, difficult_dict):
        d_question = []
        d_skill = []
        difficult_labels = []
        quest, skill = data
        data_zip = zip(quest, skill)
        for data_ in data_zip:
            d_ques, dif_label = GetPretrainDataSet.MaskedSingleDifficult(data_, difficult_dict)
            d_question.append(d_ques)
            d_skill.append(single_total_skills)
            difficult_labels.append(dif_label)
        return d_question, d_skill, difficult_labels

    @staticmethod
    def MaskedSingleDifficult(data, difficult):
        question = []
        difficult_label = []
        quest, skill = data
        assert len(quest) == len(skill)
        num = len(quest)
        for i in range(num):
            question.append(quest[i])
            difficult_label.append(float(difficult[str(quest[i])]))
        return question, difficult_label

    @staticmethod
    def maskedSingleData(data, size, mask_name):
        mask_data = []
        data_label = []
        threshold = 0.15
        num = 0
        data_ = None
        if mask_name == 'question':
            quest, skill = data
            assert len(quest) == len(skill)
            num = len(quest)
            data_ = quest
        if mask_name == 'skill':
            num = len(data)
            data_ = data
        for i in range(num):
            r = random()
            if r < threshold:  # mask 15% of all token
                if r < threshold * 0.8:  # 80% replace [mask]=size+3
                    mask_data.append(Constants.mask_Q + size)
                elif r < threshold * 0.9:  # 10% random (1, size)
                    random_q_index = randint(1, size)
                    mask_data.append(random_q_index)
                else:
                    mask_data.append(data_[i])
                data_label.append(data_[i])
            else:
                mask_data.append(data_[i])
                data_label.append(Constants.PAD)

        return mask_data, data_label


# if __name__ == '__main__':
#     training_data = torch.utils.data.DataLoader(PretrainDataSet('2009', 256, 5, 16891, 101), batch_size=128)
#     for b in training_data:
#         batch_mask_q, batch_q_label, batch_mask_s, batch_s_label, batch_question, batch_total_skill, batch_diff_label = b
#         print(batch_question[:2])
#         print(batch_total_skill[:2])
#         print()
#         break