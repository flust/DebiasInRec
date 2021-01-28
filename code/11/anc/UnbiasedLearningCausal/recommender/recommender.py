import numpy as np
import random

class Recommender(object):

    def __init__(self, num_users, num_items,
                 colname_user = 'idx_user', colname_item = 'idx_item',
                 colname_outcome = 'outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.colname_user = colname_user
        self.colname_item = colname_item
        self.colname_outcome = colname_outcome
        self.colname_prediction = colname_prediction
        self.colname_treatment = colname_treatment
        self.colname_propensity = colname_propensity

    def train(self, df, iter=100):
        pass

    def predict(self, df):
        pass

    def recommend(self, df, num_rec=10):
        pass

    def func_sigmoid(self, x):
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            return np.exp(x) / (1.0 + np.exp(x))

    def sample_time(self):
        return random.randrange(self.num_times)

    def sample_user(self, idx_time, TP=True, TN=True, CP=True, CN=True):
        while True:
            flag_condition = 1
            u = random.randrange(self.num_users)
            if TP:
                if u not in self.dict_treatment_positive_sets[idx_time]:
                    flag_condition = 0
            if TN:
                if u not in self.dict_treatment_negative_sets[idx_time]:
                    flag_condition = 0
            if CP:
                if u not in self.dict_control_positive_sets[idx_time]:
                    flag_condition = 0
            if CN:
                if u not in self.dict_control_negative_sets[idx_time]:
                    flag_condition = 0
            if flag_condition > 0:
                return u

    def sample_treatment(self, idx_time, idx_user):
        return random.choice(self.dict_treatment_sets[idx_time][idx_user])

    def sample_control(self, idx_time, idx_user):
        while True:
            flag_condition = 1
            i = random.randrange(self.num_items)
            if idx_user in self.dict_treatment_positive_sets[idx_time]:
                if i in self.dict_treatment_positive_sets[idx_time][idx_user]:
                    flag_condition = 0
            if idx_user in self.dict_treatment_negative_sets[idx_time]:
                if i in self.dict_treatment_negative_sets[idx_time][idx_user]:
                    flag_condition = 0
            if flag_condition > 0:
                return i

    # in case control is rare
    def sample_control2(self, idx_time, idx_user):
        cand_control = np.arange(self.num_items)
        cand_control = cand_control[np.isin(cand_control, self.dict_treatment_sets[idx_time][idx_user])]
        return random.choice(cand_control)

    def sample_treatment_positive(self, idx_time, idx_user):
        return random.choice(self.dict_treatment_positive_sets[idx_time][idx_user])

    def sample_treatment_negative(self, idx_time, idx_user):
        return random.choice(self.dict_treatment_negative_sets[idx_time][idx_user])

    def sample_control_positive(self, idx_time, idx_user):
        return random.choice(self.dict_control_positive_sets[idx_time][idx_user])

    def sample_control_negative(self, idx_time, idx_user):
        while True:
            flag_condition = 1
            i = random.randrange(self.num_items)
            if idx_user in self.dict_treatment_positive_sets[idx_time]:
                if i in self.dict_treatment_positive_sets[idx_time][idx_user]:
                    flag_condition = 0
            if idx_user in self.dict_treatment_negative_sets[idx_time]:
                if i in self.dict_treatment_negative_sets[idx_time][idx_user]:
                    flag_condition = 0
            if idx_user in self.dict_control_positive_sets[idx_time]:
                if i in self.dict_control_positive_sets[idx_time][idx_user]:
                    flag_condition = 0
            if flag_condition > 0:
                return i

    # TP: treatment-positive
    # CP: control-positive
    # TN: treatment-negative
    # TN: control-negative
    def sample_triplet(self):
        t = self.sample_time()
        if random.random() <= self.alpha:  # CN as positive
            if random.random() <= 0.5:  # TP as positive
                if random.random() <= 0.5:  # TP vs. TN
                    u = self.sample_user(t, TP=True, TN=True, CP=False, CN=False)
                    i = self.sample_treatment_positive(t, u)
                    j = self.sample_treatment_negative(t, u)
                else:  # TP vs. CP
                    u = self.sample_user(t, TP=True, TN=False, CP=True, CN=False)
                    i = self.sample_treatment_positive(t, u)
                    j = self.sample_control_positive(t, u)
            else:  # CN as positive
                if random.random() <= 0.5:  # CN vs. TN
                    u = self.sample_user(t, TP=False, TN=True, CP=False, CN=True)
                    i = self.sample_control_negative(t, u)
                    j = self.sample_treatment_negative(t, u)
                else:  # CN vs. CP
                    u = self.sample_user(t, TP=False, TN=False, CP=True, CN=True)
                    i = self.sample_control_negative(t, u)
                    j = self.sample_control_positive(t, u)
        else:  # CN as negative
            if random.random() <= 0.333:  # TP vs. CN
                u = self.sample_user(t, TP=True, TN=False, CP=False, CN=True)
                i = self.sample_treatment_positive(t, u)
                j = self.sample_control_negative(t, u)
            elif random.random() <= 0.5:  # TP vs. TN
                u = self.sample_user(t, TP=True, TN=True, CP=False, CN=False)
                i = self.sample_treatment_positive(t, u)
                j = self.sample_treatment_negative(t, u)
            else:  # TP vs. CP
                u = self.sample_user(t, TP=True, TN=False, CP=True, CN=False)
                i = self.sample_treatment_positive(t, u)
                j = self.sample_control_positive(t, u)

        return u, i, j

    def sample_pair(self):
        t = self.sample_time()
        if random.random() < 0.5: # pick treatment
            if random.random() > self.ratio_nega: # TP
                u = self.sample_user(t, TP=True, TN=False, CP=False, CN=False)
                i = self.sample_treatment_positive(t, u)
                flag_positive = 1
            else: # TN
                u = self.sample_user(t, TP=False, TN=True, CP=False, CN=False)
                i = self.sample_treatment_negative(t, u)
                flag_positive = 0
        else: # pick control
            if random.random() > self.ratio_nega:  # CP
                u = self.sample_user(t, TP=False, TN=False, CP=True, CN=False)
                i = self.sample_control_positive(t, u)
                flag_positive = 0
            else:  # CN
                u = self.sample_user(t, TP=False, TN=False, CP=False, CN=True)
                i = self.sample_control_negative(t, u)
                if random.random() <= self.alpha:  # CN as positive
                    flag_positive = 1
                else:
                    flag_positive = 0

        return u, i, flag_positive

    # getter
    def get_propensity(self, idx_user, idx_item):
        return self.dict_propensity[idx_user][idx_item]

