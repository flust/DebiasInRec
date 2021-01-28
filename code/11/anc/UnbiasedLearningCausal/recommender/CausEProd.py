import numpy as np
from recommender import Recommender
from numpy.random.mtrand import RandomState
import random

class CausEProd(Recommender):
    def __init__(self, num_users, num_items,
                 metric='logloss',
                 dim_factor=10, with_bias=False,
                 learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01, sd_init = 0.1,
                 reg_causal=0.01,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity)
        self.metric = metric

        self.dim_factor = dim_factor
        self.rng = RandomState(seed=None)
        self.with_bias = with_bias

        self.learn_rate = learn_rate
        self.reg_bias = reg_bias
        self.reg_factor = reg_factor
        self.reg_causal = reg_causal
        self.sd_init = sd_init
        self.flag_prepared = False

        # user_factors=user_factors_T=user_factors_C for CausE-Prod
        self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
        # item_factors_T=item_factors
        self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        self.item_factors_C = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        if self.with_bias:
            self.user_biases = np.zeros(self.num_users)
            self.item_biases = np.zeros(self.num_items)
            self.global_bias = 0.0


    def prepare_dictionary(self, df, colname_time='idx_time'):
        print("start prepare dictionary")
        self.colname_time = colname_time
        self.num_times = np.max(df.loc[:, self.colname_time]) + 1
        self.dict_treatment_positive_sets = dict()
        self.dict_treatment_negative_sets = dict()
        self.dict_treatment_sets = dict()
        self.dict_control_positive_sets = dict()
        self.dict_control_negative_sets = dict()
        # skip control_negative for its volume
        df_train = df.loc[df.loc[:, self.colname_outcome] + df.loc[:, self.colname_treatment] > 0]

        for t in np.arange(self.num_times):
            df_t = df_train.loc[df_train.loc[:, self.colname_time] == t]
            self.dict_treatment_positive_sets[t] = dict()
            self.dict_treatment_negative_sets[t] = dict()
            self.dict_treatment_sets[t] = dict()
            self.dict_control_positive_sets[t] = dict()
            self.dict_control_negative_sets[t] = dict()

            for u in np.unique(df_t.loc[:, self.colname_user]):

                df_tu = df_t.loc[df_t.loc[:, self.colname_user] == u]
                if len(df_tu) < self.num_items:  # check existence of control negatives
                    self.dict_control_negative_sets[t][u] = []

                bool_control = df_tu.loc[:, self.colname_treatment] == 0
                if np.any(bool_control):
                    self.dict_control_positive_sets[t][u] = df_tu.loc[bool_control, self.colname_item].values
                # only treatment
                bool_treatment = np.logical_not(bool_control)
                if np.any(bool_treatment):
                    df_tu = df_tu.loc[bool_treatment]
                    bool_positive = df_tu.loc[:, self.colname_outcome] > 0
                    self.dict_treatment_sets[t][u] = df_tu.loc[:, self.colname_item].values
                    if np.any(bool_positive):
                        self.dict_treatment_positive_sets[t][u] = df_tu.loc[bool_positive, self.colname_item].values
                    bool_negative = np.logical_not(bool_positive)
                    if np.any(bool_negative):
                        self.dict_treatment_negative_sets[t][u] = df_tu.loc[bool_negative, self.colname_item].values
                # else:
                #     self.dict_treatment_sets[t][u] = []

        self.flag_prepared = True
        print("prepared dictionary!")


    # override
    def sample_pair(self):
        t = self.sample_time()
        if random.random() < 0.5: # pick treatment
            flag_treatment = 1
            while True: # pick a user with treatment
                u = random.randrange(self.num_users)
                if u in self.dict_treatment_sets[t]:
                    break

            i = self.sample_treatment(t, u)
            if u in self.dict_treatment_positive_sets[t] and i in self.dict_treatment_positive_sets[t][u]:
                flag_positive = 1
            else:
                flag_positive = 0

        else: # pick control
            flag_treatment = 0
            while True: # pick a user with control
                u = random.randrange(self.num_users)
                if u in self.dict_treatment_sets[t]:
                    len_T = len(self.dict_treatment_sets[t][u])
                else:
                    len_T = 0
                if len_T < self.num_items:
                    break

            if len_T > self.num_items * 0.99:
                # print(len_T)
                i = self.sample_control2(t, u)
            else:
                i = self.sample_control(t, u)

            if u in self.dict_control_positive_sets[t] and i in self.dict_control_positive_sets[t][u]:
                flag_positive = 1
            else:
                flag_positive = 0

        return u, i, flag_positive, flag_treatment


    def train(self, df, iter = 100):

        if not self.flag_prepared: # prepare dictionary
            self.prepare_dictionary(df)

        err = 0
        current_iter = 0
        if self.metric in ['logloss']:
            while True:
                u, i, flag_positive, flag_treatment = self.sample_pair()

                u_factor = self.user_factors[u, :]
                i_factor_T = self.item_factors[i, :]
                i_factor_C = self.item_factors_C[i, :]

                if flag_treatment > 0:
                    rating = np.sum(u_factor * i_factor_T)
                else:
                    rating = np.sum(u_factor * i_factor_C)

                if self.with_bias:
                    rating += self.item_biases[i] + self.user_biases[u] + self.global_bias

                if flag_positive > 0:
                    coeff = self.func_sigmoid(-rating)
                else:
                    coeff = -self.func_sigmoid(rating)

                err += np.abs(coeff)


                i_diff_TC = i_factor_T - i_factor_C


                if flag_treatment > 0:
                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * i_factor_T - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor_T - self.reg_causal * i_diff_TC)
                    self.item_factors_C[i, :] += \
                        self.learn_rate * (self.reg_causal * i_diff_TC)
                else:
                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * i_factor_C - self.reg_factor * u_factor)
                    self.item_factors_C[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor_C + self.reg_causal * i_diff_TC)
                    self.item_factors[i, :] += \
                        self.learn_rate * (-self.reg_causal * i_diff_TC)

                if self.with_bias:
                    self.item_biases[i] += \
                        self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                    self.user_biases[u] += \
                        self.learn_rate * (coeff - self.reg_bias * self.user_biases[u])
                    self.global_bias += \
                        self.learn_rate * (coeff)

                current_iter += 1
                if current_iter >= iter:
                    return err / iter

    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred_T = np.zeros(len(df))
        pred_C = np.zeros(len(df))

        for n in np.arange(len(df)):
            pred_T[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
            if self.with_bias:
                pred_T[n] += self.item_biases[items[n]]
                pred_T[n] += self.user_biases[users[n]]
                pred_T[n] += self.global_bias
            pred_C[n] = np.inner(self.user_factors[users[n], :], self.item_factors_C[items[n], :])
            if self.with_bias:
                pred_C[n] += self.item_biases[items[n]]
                pred_C[n] += self.user_biases[users[n]]
                pred_C[n] += self.global_bias

        pred = 1 / (1 + np.exp(-pred_T)) - 1 / (1 + np.exp(-pred_C))

        return pred
