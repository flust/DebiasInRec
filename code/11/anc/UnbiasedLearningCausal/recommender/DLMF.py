import numpy as np
import pandas as pd
from recommender import Recommender
from numpy.random.mtrand import RandomState
import random

class DLMF(Recommender):
    def __init__(self, num_users, num_items,
                 metric='AR_logi', capping_T=0.01, capping_C=0.01,
                 dim_factor=200, with_bias=False, with_IPS=True,
                 only_treated=False,
                 learn_rate = 0.01, reg_factor = 0.01, reg_bias = 0.01,
                 sd_init = 0.1, reg_factor_j = 0.01, reg_bias_j = 0.01,
                 coeff_T = 1.0, coeff_C = 1.0,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction,
                         colname_treatment=colname_treatment, colname_propensity=colname_propensity)
        self.metric = metric
        self.capping_T = capping_T
        self.capping_C = capping_C
        self.with_IPS = with_IPS
        self.dim_factor = dim_factor
        self.rng = RandomState(seed=None)
        self.with_bias = with_bias
        self.coeff_T = coeff_T
        self.coeff_C = coeff_C
        self.learn_rate = learn_rate
        self.reg_bias = reg_bias
        self.reg_factor = reg_factor
        self.reg_bias_j = reg_bias_j
        self.reg_factor_j = reg_factor_j
        self.sd_init = sd_init
        self.only_treated = only_treated

        self.user_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_users, self.dim_factor))
        self.item_factors = self.rng.normal(loc=0, scale=self.sd_init, size=(self.num_items, self.dim_factor))
        if self.with_bias:
            self.user_biases = np.zeros(self.num_users)
            self.item_biases = np.zeros(self.num_items)
            self.global_bias = 0.0

    def train(self, df, iter = 100):
        df_train = df.loc[df.loc[:, self.colname_outcome] > 0, :] # need only positive outcomes
        if self.only_treated: # train only with treated positive (DLTO)
            df_train = df_train.loc[df_train.loc[:, self.colname_treatment] > 0, :]

        if self.capping_T is not None:
            bool_cap = np.logical_and(df_train.loc[:, self.colname_propensity] < self.capping_T, df_train.loc[:, self.colname_treatment] == 1)
            if np.sum(bool_cap) > 0:
                df_train.loc[bool_cap, self.colname_propensity] = self.capping_T
        if self.capping_C is not None:      
            bool_cap = np.logical_and(df_train.loc[:, self.colname_propensity] > 1 - self.capping_C, df_train.loc[:, self.colname_treatment] == 0)
            if np.sum(bool_cap) > 0:
                df_train.loc[bool_cap, self.colname_propensity] = 1 - self.capping_C

        if self.with_IPS: # point estimate of individual treatment effect (ITE) <- for binary outcome abs(ITE) = IPS
            df_train.loc[:, 'ITE'] =  df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]/df_train.loc[:, self.colname_propensity] - \
                                      (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]/(1 - df_train.loc[:, self.colname_propensity])

        else:
            df_train.loc[:, 'ITE'] =  df_train.loc[:, self.colname_treatment] * df_train.loc[:, self.colname_outcome]  - \
                                      (1 - df_train.loc[:, self.colname_treatment]) * df_train.loc[:, self.colname_outcome]

        err = 0
        current_iter = 0
        while True:
            df_train = df_train.sample(frac=1)
            users = df_train.loc[:, self.colname_user].values
            items = df_train.loc[:, self.colname_item].values
            ITE = df_train.loc[:, 'ITE'].values

            if self.metric in ['AR_logi', 'AR_sig', 'AR_hinge']:
                for n in np.arange(len(df_train)):

                    u = users[n]
                    i = items[n]

                    while True:
                        j = random.randrange(self.num_items)
                        if i != j:
                            break

                    u_factor = self.user_factors[u, :]
                    i_factor = self.item_factors[i, :]
                    j_factor = self.item_factors[j, :]

                    diff_rating = np.sum(u_factor * (i_factor - j_factor))
                    if self.with_bias:
                        diff_rating += (self.item_biases[i] - self.item_biases[j])

                    if self.metric == 'AR_logi':
                        if ITE[n] >= 0:
                            coeff = ITE[n] * self.coeff_T * self.func_sigmoid(-self.coeff_T * diff_rating) # Z=1, Y=1
                        else:
                            coeff = ITE[n] * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating) # Z=0, Y=1

                    elif self.metric == 'AR_sig':
                        if ITE[n] >= 0:
                            coeff = ITE[n] * self.coeff_T * self.func_sigmoid(self.coeff_T * diff_rating) * self.func_sigmoid(-self.coeff_T * diff_rating)
                        else:
                            coeff = ITE[n] * self.coeff_C * self.func_sigmoid(self.coeff_C * diff_rating) * self.func_sigmoid(-self.coeff_C * diff_rating)

                    elif self.metric == 'AR_hinge':
                        if ITE[n] >= 0:
                            if self.coeff_T > 0 and diff_rating < 1.0/self.coeff_T:
                                coeff = ITE[n] * self.coeff_T 
                            else:
                                coeff = 0.0
                        else:
                            if self.coeff_C > 0 and diff_rating > -1.0/self.coeff_C:
                                coeff = ITE[n] * self.coeff_C
                            else:
                                coeff = 0.0

                    err += np.abs(coeff)

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
