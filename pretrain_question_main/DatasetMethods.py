class DatasetMethods:

    def __init__(self, data_name, sq_len, step):
        self.train_file_path = '../data_question/' + data_name + '/assist' + data_name + '_pid_train.csv'
        self.difficult_path = '../data_question/' + data_name + '/diffict_' + data_name + '.txt'
        self.total_skill_path = '../data_question/' + data_name + '/assist' + data_name + '_train_skill.csv'
        self.single_total_skill_path = '../data_question/' + data_name + '/assist' + data_name + '_skill.csv'
        self.sq_len = sq_len
        self.step = step

    @staticmethod
    def readFile(path):
        with open(path) as f:
            data = f.readlines()
        f.close()
        return data

    def getTrainDataAndSkillDataAndDifficultDict(self):
        train_data = self.readDataSet()
        shuffle_total_skills = self.readShuffleTotalSkill()
        single_total_skills = self.readSequenceSkill()
        difficult_dict = self.readDifficultDict()
        return train_data, shuffle_total_skills, single_total_skills, difficult_dict

    def readDataSet(self):
        data = DatasetMethods.readFile(self.train_file_path)
        questions = []
        skills = []
        cycle = int(len(data) / 4)
        for i in range(cycle):
            num = int(data[4 * i].replace('\n', ''))
            list_q = data[4 * i + 1].replace('\n', '').split(',')
            list_q = list(map(lambda x: float(x), list_q))
            list_s = data[4 * i + 2].replace('\n', '').split(',')
            list_s = list(map(lambda x: float(x), list_s))
            if num >= self.sq_len:
                if num == self.sq_len:
                    questions.append(list_q)
                    skills.append(list_s)
                else:
                    mod = num % self.sq_len
                    cycle_m = int(num / self.sq_len) - 1
                    windows = mod + self.sq_len * cycle_m
                    for i in range(windows + 1):
                        index = self.step * i
                        if index > windows:
                            break
                        list_qu = list_q[index:index + self.sq_len]
                        list_sk = list_s[index:index + self.sq_len]
                        questions.append(list_qu)
                        skills.append(list_sk)

        return questions, skills

    def readShuffleTotalSkill(self):
        data = DatasetMethods.readFile(self.total_skill_path)
        total_skills = []
        for i in range(len(data)):
            list_s = data[i].replace('\n', '').split(',')
            list_s = list(map(lambda x: float(x), list_s))
            total_skills.append(list_s)
        return total_skills

    def readSequenceSkill(self):
        data = DatasetMethods.readFile(self.single_total_skill_path)
        single_total_skill = []
        for i in data:
            single_total_skill = i.replace('\n', '').split(',')
            single_total_skill = list(map(lambda x: float(x), single_total_skill))
        return single_total_skill

    def readDifficultDict(self):
        data = DatasetMethods.readFile(self.difficult_path)
        list_q = data[0].replace('\n', '').split(',')
        list_difficult = data[1].replace('\n', '').split(',')
        assert len(list_q) == len(list_difficult)
        difficult_dict = {}
        for i in range(len(list_q)):
            difficult_dict[list_q[i]] = list_difficult[i]
        return difficult_dict
