
import numpy as np
import pandas as pd
from recommender import RandomBase, PopularBase, NeighborBase, LMF

class DataGenerator():
    def __init__(self, rate_prior=0.1,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome',
                 colname_outcome_treated='outcome_T', colname_outcome_control='outcome_C',
                 colname_treatment='treated', colname_propensity='propensity',
                 colname_effect='causal_effect', colname_expectation='causal_effect_expectation',
                 colname_prediction='pred',
                 random_seed=1):
        self.rate_prior = rate_prior
        self.colname_user = colname_user
        self.colname_item = colname_item
        self.colname_outcome = colname_outcome
        self.colname_outcome_treated = colname_outcome_treated
        self.colname_outcome_control = colname_outcome_control
        self.colname_effect = colname_effect
        self.colname_expectation = colname_expectation
        self.colname_treatment = colname_treatment
        self.colname_propensity = colname_propensity
        self.colname_prediction = colname_prediction
        self.random_seed = random_seed
        self.optimal_power = None

    def load_data(self, path_empirical_data):
        self.path_empirical_data = path_empirical_data
        self.df_data = pd.read_csv(self.path_empirical_data)

        self.df_data.loc[:, 'num_control'] = self.df_data.loc[:, 'num_visit'] - self.df_data.loc[:, 'num_treatment']
        self.df_data.loc[:, 'num_control_outcome'] = self.df_data.loc[:, 'num_outcome'] - self.df_data.loc[:, 'num_treated_outcome']

        # get item means
        df_mean = self.df_data.loc[:, [self.colname_item, 'num_treated_outcome', 'num_control_outcome',
                                       'num_treatment', 'num_control', 'num_outcome', 'num_visit']]
        df_mean = df_mean.groupby(self.colname_item, as_index=False).mean()
        df_mean = df_mean.rename(columns={'num_treated_outcome': 'num_treated_outcome_mean',
                                          'num_control_outcome': 'num_control_outcome_mean',
                                          'num_treatment': 'num_treatment_mean',
                                          'num_control': 'num_control_mean',
                                          'num_outcome': 'num_outcome_mean',
                                          'num_visit': 'num_visit_mean'})
        # merge
        self.df_data = pd.merge(self.df_data, df_mean, on=[self.colname_item], how='left')

        self.df_data.loc[:, 'prob_outcome_treated'] = \
            (self.df_data.loc[:, 'num_treated_outcome'] + self.rate_prior * self.df_data.loc[:, 'num_treated_outcome_mean']) / \
            (self.df_data.loc[:, 'num_treatment'] + self.rate_prior * self.df_data.loc[:, 'num_treatment_mean'])
        self.df_data.loc[:, 'prob_outcome_control'] = \
            (self.df_data.loc[:, 'num_control_outcome'] + self.rate_prior * self.df_data.loc[:,'num_control_outcome_mean']) / \
            (self.df_data.loc[:, 'num_control'] + self.rate_prior * self.df_data.loc[:, 'num_control_mean'])
        self.df_data.loc[:, 'prob_outcome'] = \
            (self.df_data.loc[:, 'num_outcome'] + self.rate_prior * self.df_data.loc[:, 'num_outcome_mean']) / \
            (self.df_data.loc[:, 'num_visit'] + self.rate_prior * self.df_data.loc[:, 'num_visit_mean'])

        self.num_data = self.df_data.shape[0]
        self.num_users = np.max(self.df_data.loc[:, self.colname_user].values) + 1
        self.num_items = np.max(self.df_data.loc[:, self.colname_item].values) + 1

    def get_optimal_power(self, num_rec):
        n = num_rec
        while True:
            temp_propensity = 1 - np.power(1 - self.df_data.loc[:, self.colname_propensity], n)
            if round(np.sum(temp_propensity) / self.num_users) == num_rec:
                self.optimal_power = n
                break
            else:
                n += 1

    def assign_propensity(self, capping = 0.01, mode='original', scale_factor=1.0, num_rec=100, df_train=None):

        if mode == 'original':
            self.df_data.loc[:, self.colname_propensity] = \
                (self.df_data.loc[:, 'num_treatment'] + self.rate_prior * self.df_data.loc[:,'num_treatment_mean']) / \
                (self.df_data.loc[:, 'num_visit'] + self.rate_prior * self.df_data.loc[:, 'num_visit_mean'])
            self.df_data.loc[:, self.colname_prediction] = 0.0

        elif mode in ['pref', 'prefT', 'prefC']:
            df = self.df_data
            if mode == 'pref':
                df.loc[:, self.colname_propensity] = df.loc[:, 'prob_outcome']
            elif mode == 'prefT':
                df.loc[:, self.colname_propensity] = df.loc[:, 'prob_outcome_treated']
            elif mode == 'prefC':
                df.loc[:, self.colname_propensity] = df.loc[:, 'prob_outcome_control']

            df.loc[:, self.colname_propensity] = np.power(df.loc[:, self.colname_propensity], scale_factor)
            while True:
                df.loc[df.loc[:, self.colname_propensity] > 1, self.colname_propensity] = 1.0
                total_num_rec = np.sum(df.loc[:, self.colname_propensity])
                avg_num_rec = total_num_rec/self.num_users
                print(avg_num_rec)
                if round(avg_num_rec) != num_rec:
                    df.loc[:, self.colname_propensity] = df.loc[:, self.colname_propensity] * num_rec/avg_num_rec
                else:
                    break
            self.df_data = df

        else:
            if '_' in mode:
                mode, type_recommender = mode.split('_')
            else:
                if mode[-1] == 'C':
                    type_recommender = 'oracleC'
                elif mode[-1] == 'T':
                    type_recommender = 'oracleT'
                else:
                    type_recommender = 'oracle'

            print('type_recommender: ' + type_recommender)
            df = self.calc_score(df_train, self.df_data, type_recommender=type_recommender)
            # get ranking
            df = df.sort_values(by=[self.colname_user, self.colname_prediction], ascending=False)
            df.loc[:, 'rank'] = np.repeat(np.arange(self.num_items) + 1, self.num_users)

            # scaling
            if mode in ['rank', 'rankC', 'rankT']:
                df.loc[:, self.colname_propensity] = 1.0 / np.power(df.loc[:, 'rank'], scale_factor)
                sum_propensity = np.sum(1.0 / np.power(np.arange(self.num_items) + 1, scale_factor))
            elif mode in ['logrank', 'logrankC', 'logrankT']:
                df.loc[:, self.colname_propensity] = 1.0 / np.power(np.log2(df.loc[:, 'rank'] + 1), scale_factor)
                sum_propensity = np.sum(1.0 / np.power(np.log2(np.arange(self.num_items) + 2), scale_factor))

            df.loc[:, self.colname_propensity] /= sum_propensity
            df.loc[:, self.colname_propensity] *= num_rec

            while True:
                df.loc[df.loc[:, self.colname_propensity] > 1, self.colname_propensity] = 1.0
                total_num_rec = np.sum(df.loc[:, self.colname_propensity])
                avg_num_rec = total_num_rec/self.num_users
                print(avg_num_rec)
                if round(avg_num_rec) < num_rec:
                    df.loc[:, self.colname_propensity] = df.loc[:, self.colname_propensity] * num_rec/avg_num_rec
                else:
                    break
            self.df_data = df

        if capping is not None:
            self.df_data.loc[self.df_data.loc[:, self.colname_propensity] < capping, self.colname_propensity] = capping
            self.df_data.loc[self.df_data.loc[:, self.colname_propensity] > 1 - capping, self.colname_propensity] = 1 - capping

    def assign_treatment(self):
        self.df_data.loc[:, self.colname_treatment] = 0
        bool_treatment = self.df_data.loc[:, self.colname_propensity] > np.random.rand(self.num_data)
        self.df_data.loc[bool_treatment, self.colname_treatment] = 1

    def assign_outcome(self):
        self.df_data.loc[:, self.colname_outcome] = 0
        prob = np.random.rand(self.num_data)
        self.df_data.loc[:, self.colname_outcome_treated] = (self.df_data.loc[:, 'prob_outcome_treated'] >= prob) * 1.0
        prob = np.random.rand(self.num_data)
        self.df_data.loc[:, self.colname_outcome_control] = (self.df_data.loc[:, 'prob_outcome_control'] >= prob) * 1.0

        self.df_data.loc[:, self.colname_outcome] = \
            self.df_data.loc[:, self.colname_treatment] * self.df_data.loc[:, self.colname_outcome_treated] + \
            (1 - self.df_data.loc[:, self.colname_treatment]) * self.df_data.loc[:, self.colname_outcome_control]

    def assign_effect(self):
        self.df_data.loc[:, self.colname_effect] = \
            self.df_data.loc[:, self.colname_outcome_treated] - self.df_data.loc[:,self.colname_outcome_control]
        self.df_data.loc[:, self.colname_expectation] = \
            self.df_data.loc[:, 'prob_outcome_treated'] - self.df_data.loc[:, 'prob_outcome_control']

    def get_observation(self, with_additional_info=False):
        if with_additional_info:
            return self.df_data.loc[:,
                   [self.colname_user, self.colname_item, self.colname_treatment, self.colname_outcome, self.colname_propensity,
                    self.colname_effect, self.colname_expectation, self.colname_prediction,
                    'prob_outcome_treated', 'prob_outcome_control', 'prob_outcome']]
        else:
            return self.df_data.loc[:, [self.colname_user, self.colname_item, self.colname_treatment, self.colname_outcome, self.colname_propensity, self.colname_effect]]

    def get_groundtruth(self):
        return self.df_data.loc[:, [self.colname_user, self.colname_item, self.colname_effect]]

    def add_true_causal_effect(self, df_data):
        df_data_causal_effect = self.df_data.copy()
        df_data_causal_effect = df_data_causal_effect.loc[:, [self.colname_user, self.colname_item, self.colname_effect]]
        df_data_causal_effect = df_data_causal_effect.drop_duplicates()

        df_data = pd.merge(df_data, df_data_causal_effect, on=[self.colname_user, self.colname_item], how='left')
        return df_data


    def calc_score(self, df_train, df_pred, type_recommender='kNN'):
        if type_recommender == 'kNN':
            recommender = NeighborBase(num_users=self.num_users, num_items=self.num_items,
                                           colname_user=self.colname_user, colname_item=self.colname_item,
                                           colname_outcome=self.colname_outcome, colname_prediction=self.colname_prediction,
                                           measure_simil='cosine', way_neighbor='user', num_neighbor=100)
            recommender.train(df_train, iter=1)
            df_pred.loc[:, self.colname_prediction] = recommender.predict(df_pred)
            df_pred.loc[:, self.colname_prediction] += 0.0000000001 * np.random.rand(len(df_pred))

        elif type_recommender == 'BPR':
            recommender = LMF(num_users=self.num_users, num_items=self.num_items,
                              colname_user=self.colname_user, colname_item=self.colname_item,
                              colname_outcome=self.colname_outcome, colname_prediction=self.colname_prediction,
                              dim_factor=200, with_bias=False,
                              learn_rate=0.1,
                              sd_init=0.1 / np.sqrt(200),
                              reg_factor=0.1, reg_bias=0.1,
                              metric='AUC', ratio_nega=0.5)
            recommender.train(df_train, iter=100 *1000000)
            df_pred.loc[:, self.colname_prediction] = recommender.predict(df_pred)
        elif type_recommender == 'oracle':
            df_pred.loc[:, self.colname_prediction] = df_pred.loc[:, 'prob_outcome']
        elif type_recommender == 'oracleC':
            df_pred.loc[:, self.colname_prediction] = df_pred.loc[:, 'prob_outcome_control']
        elif type_recommender == 'oracleT':
            df_pred.loc[:, self.colname_prediction] = df_pred.loc[:, 'prob_outcome_treated']
        return df_pred