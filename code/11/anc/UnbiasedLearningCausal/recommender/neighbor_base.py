import numpy as np
from recommender import Recommender

# for binary outcome
class NeighborBase(Recommender):

    def __init__(self, num_users, num_items,
                 colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 measure_simil='cosine', way_neighbor='user', num_neighbor=100,
                 scale_similarity=1.0, normalize_similarity=False):

        super().__init__(num_users=num_users, num_items=num_items,
                         colname_user=colname_user, colname_item=colname_item,
                         colname_outcome=colname_outcome, colname_prediction=colname_prediction)
        self.measure_simil = measure_simil
        self.way_neighbor = way_neighbor
        self.num_neighbor = num_neighbor
        self.scale_similarity = scale_similarity
        self.normalize_similarity = normalize_similarity

    def simil(self, set1, set2, measure_simil):
        if measure_simil == "jaccard":
            return self.simil_jaccard(set1, set2)
        elif measure_simil == "cosine":
            return self.simil_cosine(set1, set2)

    def train(self, df, iter=1):
        df_posi = df.loc[df.loc[:, self.colname_outcome] > 0]

        dict_items2users = dict() # map an item to users who consumed the item
        for i in np.arange(self.num_items):
            dict_items2users[i] = np.unique(df_posi.loc[df_posi.loc[:, self.colname_item] == i, self.colname_user].values)
        self.dict_items2users = dict_items2users
        print("prepared dict_item2users")

        dict_users2items = dict()  # map an user to items which are consumed by the user
        for u in np.arange(self.num_users):
            dict_users2items[u] = np.unique(df_posi.loc[df_posi.loc[:, self.colname_user] == u, self.colname_item].values)
        self.dict_users2items = dict_users2items
        print("prepared dict_users2items")

        if self.way_neighbor == 'user':
            dict_simil_users = {}
            for u1 in np.arange(self.num_users):
                if u1 % 1000 == 0:
                    print("progress of similarity computation: {:.1f} %".format(100 * u1/self.num_users))

                items_u1 = self.dict_users2items[u1]
                dict_neighbor = {}
                if len(items_u1) > 0:
                    cand_u2 = np.unique(df_posi.loc[np.isin(df_posi.loc[:, self.colname_item], items_u1), self.colname_user].values)
                    for u2 in cand_u2:
                        if u2 != u1:
                            items_u2 = self.dict_users2items[u2]
                            dict_neighbor[u2] = self.simil(items_u1, items_u2, self.measure_simil)

                    if len(dict_neighbor) > self.num_neighbor:
                        dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
                    if self.scale_similarity != 1.0:
                        dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
                    if self.normalize_similarity:
                        dict_neighbor = self.normalize_neighbor(dict_neighbor)
                    dict_simil_users[u1] = dict_neighbor
                else:
                    dict_simil_users[u1] = dict_neighbor
            self.dict_simil_users = dict_simil_users

        elif self.way_neighbor == 'item':
            dict_simil_items = {}
            for i1 in np.arange(self.num_items):
                if i1 % 1000 == 0:
                    print("progress of similarity computation: {:.1f} %".format(100 * i1 / self.num_items))

                users_i1 = self.dict_items2users[i1]
                dict_neighbor = {}
                if len(users_i1) > 0:
                    cand_i2 = np.unique(
                        df_posi.loc[np.isin(df_posi.loc[:, self.colname_user], users_i1), self.colname_item].values)
                    for i2 in cand_i2:
                        if i2 != i1:
                            users_i2 = self.dict_items2users[i2]
                            dict_neighbor[i2] = self.simil(users_i1, users_i2, self.measure_simil)

                    if len(dict_neighbor) > self.num_neighbor:
                        dict_neighbor = self.trim_neighbor(dict_neighbor, self.num_neighbor)
                    if self.scale_similarity != 1.0:
                        dict_neighbor = self.rescale_neighbor(dict_neighbor, self.scale_similarity)
                    if self.normalize_similarity:
                        dict_neighbor = self.normalize_neighbor(dict_neighbor)
                    dict_simil_items[i1] = dict_neighbor
                else:
                    dict_simil_items[i1] = dict_neighbor
            self.dict_simil_items = dict_simil_items

    def trim_neighbor(self, dict_neighbor, num_neighbor):
        return dict(sorted(dict_neighbor.items(), key=lambda x:x[1], reverse = True)[:num_neighbor])

    def normalize_neighbor(self, dict_neighbor):
        sum_simil = 0.0
        for v in dict_neighbor.values():
            sum_simil += v
        for k, v in dict_neighbor.items():
            dict_neighbor[k] = v/sum_simil
        return dict_neighbor

    def rescale_neighbor(self, dict_neighbor, scaling_similarity=1.0):
        for k, v in dict_neighbor.items():
            dict_neighbor[k] = np.power(v, scaling_similarity)
        return dict_neighbor

    def predict(self, df):
        users = df[self.colname_user].values
        items = df[self.colname_item].values
        pred = np.zeros(len(df))
        if self.way_neighbor == 'user':
            for n in np.arange(len(df)):
                i_users = self.dict_items2users[items[n]] # users who consumed i=items[n]
                score = 0.0
                if len(i_users) > 0:
                    u1 = users[n]
                    for u2 in i_users:
                        if u2 in self.dict_simil_users[u1].keys():
                            score += self.dict_simil_users[u1][u2]

                pred[n] = score

        elif self.way_neighbor == 'item':
            for n in np.arange(len(df)):
                u_items = self.dict_users2items[users[n]] # items that is consumed by u=users[n]
                score = 0.0
                if len(u_items) > 0:
                    i1 = items[n]
                    for i2 in u_items:
                        if i2 in self.dict_simil_items[i1].keys():
                            score += self.dict_simil_items[i1][i2]

                pred[n] = score

        return pred

    def simil_jaccard(self, x, y):
        return len(np.intersect1d(x, y))/len(np.union1d(x, y))

    def simil_cosine(self, x, y):
        return len(np.intersect1d(x, y))/np.sqrt(len(x)*len(y))

