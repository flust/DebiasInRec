import numpy as np
from recommender import Recommender
from numpy.random.mtrand import RandomState
import random

class ULMF(Recommender):
    def __init__(self, num_users, num_items,
                 metric='AUC', ratio_nega=0.8, alpha=0.0,
                 dim_factor=200, with_bias=False,
                 learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01, sd_init = 0.1,
                 reg_factor_j=0.01, reg_bias_j=0.01,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity)
        self.metric = metric
        self.ratio_nega = ratio_nega

        self.alpha = alpha
        self.dim_factor = dim_factor
        self.rng = RandomState(seed=None)
        self.with_bias = with_bias
        self.sd_init = sd_init

        self.learn_rate = learn_rate
        self.reg_bias = reg_bias
        self.reg_factor = reg_factor
        self.reg_bias_j = reg_bias_j
        self.reg_factor_j = reg_factor_j
        self.flag_prepared = False

        self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
        self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
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
        self.dict_control_positive_sets = dict()
        self.dict_control_negative_sets = dict()
        # remove control_negative to save memory usage
        df_train = df.loc[df.loc[:, self.colname_outcome] + df.loc[:, self.colname_treatment] > 0]

        for t in np.arange(self.num_times):
            df_t = df_train.loc[df_train.loc[:, self.colname_time] == t]
            self.dict_treatment_positive_sets[t] = dict()
            self.dict_treatment_negative_sets[t] = dict()
            self.dict_control_positive_sets[t] = dict()
            self.dict_control_negative_sets[t] = dict()

            for u in np.unique(df_t.loc[:, self.colname_user]):
                df_tu = df_t.loc[df_t.loc[:, self.colname_user] == u]
                if len(df_tu) < self.num_items: # check existence of control negatives
                    self.dict_control_negative_sets[t][u] = []

                bool_control = df_tu.loc[:, self.colname_treatment] == 0
                if np.any(bool_control):
                    self.dict_control_positive_sets[t][u] = df_tu.loc[bool_control, self.colname_item].values
                # only treatment
                bool_treatment = np.logical_not(bool_control)
                if np.any(bool_treatment):
                    df_tu = df_tu.loc[bool_treatment]
                    bool_positive = df_tu.loc[:, self.colname_outcome] > 0
                    if np.any(bool_positive):
                        self.dict_treatment_positive_sets[t][u] = df_tu.loc[bool_positive, self.colname_item].values
                    bool_negative = np.logical_not(bool_positive)
                    if np.any(bool_negative):
                        self.dict_treatment_negative_sets[t][u] = df_tu.loc[bool_negative, self.colname_item].values

        self.flag_prepared = True
        print("prepared dictionary!")


    def train(self, df, iter = 100):

        if not self.flag_prepared: # prepare dictionary
            self.prepare_dictionary(df)

        err = 0
        current_iter = 0
        while True:
            if self.metric == 'AUC': # ULBPR
                while True:
                    u, i, j = self.sample_triplet()

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]
                    j_factor = self.item_factors[j, :]

                    diff_rating = np.sum(u_factor * (i_factor - j_factor))

                    if self.with_bias:
                        diff_rating += (self.item_biases[i] - self.item_biases[j])

                    coeff = self.func_sigmoid(-diff_rating)

                    err += coeff

                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * (i_factor - j_factor) - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)
                    self.item_factors[j, :] += \
                        self.learn_rate * (-coeff * u_factor - self.reg_factor_j * j_factor)

                    if self.with_bias:
                        self.item_biases[i] += \
                            self.learn_rate * (coeff - self.reg_bias * self.item_biases[i])
                        self.item_biases[j] += \
                            self.learn_rate * (-coeff - self.reg_bias_j * self.item_biases[j])

                    current_iter += 1
                    if current_iter >= iter:
                        return err/iter

            elif self.metric == 'logloss': # ULRMF
                while True:
                    u, i, flag_positive = self.sample_pair()

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]

                    rating = np.sum(u_factor * i_factor)

                    if self.with_bias:
                        rating += self.item_biases[i] + self.user_biases[u] + self.global_bias

                    if flag_positive > 0:
                        coeff = self.func_sigmoid(-rating)
                    else:
                        coeff = -self.func_sigmoid(rating)

                    err += np.abs(coeff)

                    self.user_factors[u, :] += \
                        self.learn_rate * (coeff * i_factor - self.reg_factor * u_factor)
                    self.item_factors[i, :] += \
                        self.learn_rate * (coeff * u_factor - self.reg_factor * i_factor)

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
        pred = np.zeros(len(df))
        for n in np.arange(len(df)):
            pred[n] = np.inner(self.user_factors[users[n], :], self.item_factors[items[n], :])
            if self.with_bias:
                pred[n] += self.item_biases[items[n]]
                pred[n] += self.user_biases[users[n]]
                pred[n] += self.global_bias

        # pred = 1 / (1 + np.exp(-pred))
        return pred
