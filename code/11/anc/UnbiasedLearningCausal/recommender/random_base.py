import numpy as np
from recommender import Recommender

class RandomBase(Recommender):

    def __init__(self, num_users, num_items,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred'):
        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction)

    def train(self, df, iter = 1):
        pass

    def predict(self, df):
        return np.random.rand(df.shape[0])

