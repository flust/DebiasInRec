import numpy as np
import pandas as pd
from recommender import Recommender

class PopularBase(Recommender):

    def __init__(self, num_users, num_items,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction)

    def train(self, df, iter = 1):
        df_cnt = df.groupby(self.colname_item, as_index=False)[self.colname_outcome].sum()
        df_cnt['prob'] = df_cnt[self.colname_outcome] /self.num_users
        self.df_cnt = df_cnt

    def predict(self, df):
        df = pd.merge(df, self.df_cnt, on=self.colname_item, how='left')
        return df.loc[:, 'prob'].values

