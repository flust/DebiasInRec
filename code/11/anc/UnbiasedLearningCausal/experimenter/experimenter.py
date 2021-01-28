import numpy as np
import pandas as pd
from datetime import datetime
from evaluator import Evaluator
from recommender import RandomBase, PopularBase, NeighborBase, LMF, ULMF, DLMF, CausE, CausEProd

class Experimenter():
    def __init__(self, colname_user='idx_user', colname_item='idx_item',
                 colname_outcome='outcome', colname_prediction='pred',
                 colname_treatment='treated', colname_propensity='propensity'):
        self.colname_user = colname_user
        self.colname_item = colname_item
        self.colname_outcome = colname_outcome
        self.colname_prediction = colname_prediction
        self.colname_treatment = colname_treatment
        self.colname_propensity = colname_propensity

        self.monitor = ['AUC', 'CPrec_10', 'CPrec_100', 'CDCG_100000']

        self.evaluator = Evaluator()
        self.default_params = {
            # model condition
            'recommender': 'DLMF',
            'dim_factor': 200,
            'with_bias': False, # this 'bias' means bias terms in MF, irrelevant to the bias discussed in the paper.

            # train condition
            'reg_common': 0.01, # all regs are set to the value of reg_common if not specified.
            'sd_init': 0.1,
            'reg_factor': -1.,
            'reg_bias': -1.,
            'reg_factor_j': -1.,
            'reg_bias_j': -1.,
            'reg_causal': -1.,
            'learn_rate': 0.01,
            'train_metric': 'AR_logi',
            'capping': 0.001, # Both cappings are set to the value of capping if not specified.
            'capping_T': -1.,
            'capping_C': -1.,
            'naive': False,
            'with_IPS': True,
            'ratio_nega': 0.8,
            'ratio_treatment': 0.5,
            'coeff_T': -1.0,
            'coeff_C': -1.0,
            'coeff_common': 1.0, # Both coeffs are set to the value of coeff_common if not specified. (not used in the paper.)
            'only_treated': False,
            'alpha': 0.0, # alpha for ULBPR and ULRMF

            # kNN condition (not used in the paper)
            'measure_simil': 'cosine',
            'way_simil': 'treatment',
            'way_neighbor': 'user',
            'num_neighbor': 100,
            'scale_similarity': 1.0,
            'normalize_similarity': False,
            'convert_count': 'log2',

            # mis-specification of IPS
            'mis_IPS_train' : 0.0,
            'mis_IPS_eval' : 0.0,

            # eval condition
            'num_loop': 1,
            'interval_eval': 1,
            'eval_train_data': False,
            'check_factors': True,

            'eval_metrics': 'Prec_10-Prec_100-AUC' + '-CPrec_10-CPrec_100-CDCG_100000-CAR'
        }

    def fill_defaults(self, params):
        for k, v in self.default_params.items():
            if k in params.keys():
                if not type(self.default_params[k]) == type(params[k]):
                    if type(self.default_params[k]) == int:
                        params[k] = int(params[k])
                    elif type(self.default_params[k]) == float:
                        params[k] = float(params[k])
                    elif type(self.default_params[k]) == bool:
                        if type(params[k]) == str:
                            if params[k][0].upper() == 'T':
                                params[k] = True
                            else:
                                params[k] = False
                        elif type(params[k]) == int or type(params[k]) == float:
                            if params[k] > 0:
                                params[k] = True
                            else:
                                params[k] = False
            else:
                params[k] = v

        return params


    def validate(self, recommender, df_train, df_vali, num_loop, interval_eval, eval_metrics, eval_train_data=False, check_factors=False):
        print("eval_metrics: {}".format(eval_metrics))

        dict_result = dict()
        dict_result['iter'] = []
        dict_result['t_train'] = []
        dict_result['t_pred'] = []
        dict_result['t_eval'] = []

        unique_time = np.unique(df_vali.loc[:, 'idx_time'].values)
        print('len(unique_time): {}'.format(len(unique_time)))
        if len(unique_time) == 1:
            for eval_metric in eval_metrics.split('-'):
                dict_result[eval_metric] = []
        else:
            for eval_metric in eval_metrics.split('-'):
                for m in unique_time:
                    dict_result['ter' + str(m) + '_' + eval_metric] = []

        if eval_train_data:
            for eval_metric in eval_metrics.split('-'):
                dict_result[eval_metric+'_tr'] = []

        if check_factors:
            dict_result['std_user_factors'] = []
            dict_result['std_item_factors'] = []
            dict_result['std_item_biases'] = []

        print("start training.")
        for n in np.arange(num_loop):

            t_init = datetime.now()
            avg_err = recommender.train(df_train, iter = interval_eval)
            print("avg_err: {}".format(avg_err))
            t_diff = (datetime.now() - t_init)
            t_train = t_diff.seconds / 60

            t_init = datetime.now()
            df_vali.loc[:, self.colname_prediction] = recommender.predict(df_vali)
            t_diff = (datetime.now() - t_init)
            t_pred = t_diff.seconds / 60

            t_init = datetime.now()
            for eval_metric in eval_metrics.split('-'):
                if 'AUC' in eval_metric or 'AR' in eval_metric:
                    measure = eval_metric
                    num_rec = np.nan
                else:
                    measure, num_rec = eval_metric.split('_')
                    num_rec = int(num_rec)

                mode = ''
                if measure[:6] == 'uplift':
                    meas, mode, cap_prop = measure.split('in')
                    cap_prop = float(cap_prop)
                elif 'IPS' in measure:
                    meas, cap_prop = measure.split('in')
                    cap_prop = float(cap_prop)
                else:
                    meas = measure
                    cap_prop = np.nan

                if len(unique_time) == 1:
                    res = self.evaluator.evaluate(df_vali, measure=meas, num_rec=num_rec, mode=mode, cap_prop=cap_prop)
                    if eval_metric in self.monitor:
                        print("{}: {}".format(eval_metric, res))
                    dict_result[eval_metric].append(res)
                else:
                    for m in unique_time:
                        df_vali_m = df_vali.loc[df_vali.loc[:, 'idx_time'] == m, :]
                        res = self.evaluator.evaluate(df_vali_m, measure=meas, num_rec=num_rec, cap_prop=cap_prop)
                        dict_result['ter' + str(m) + '_' + eval_metric].append(res)  # ter = TEst Round
                    print("Finished {} with {} round".format(eval_metric, len(unique_time)))


            if check_factors:
                if 'user_factors' in recommender.__dict__.keys():
                    dict_result['std_user_factors'].append(float(np.std(recommender.user_factors)))
                    dict_result['std_item_factors'].append(float(np.std(recommender.item_factors)))
                else:
                    dict_result['std_user_factors'].append(0.0)
                    dict_result['std_item_factors'].append(0.0)
                if 'item_biases' in recommender.__dict__.keys():
                    # dict_result['std_user_factors'].append(np.std(recommender.user_factors))
                    # dict_result['std_item_factors'].append(np.std(recommender.item_factors))
                    dict_result['std_item_biases'].append(float(np.std(recommender.item_biases)))
                else:
                    dict_result['std_item_biases'].append(0.0)

            t_diff = (datetime.now() - t_init)
            t_eval = t_diff.seconds / 60
            dict_result['iter'].append((n + 1) * interval_eval)
            dict_result['t_train'].append(t_train)
            dict_result['t_pred'].append(t_pred)
            dict_result['t_eval'].append(t_eval)

            print("iter: {}".format((n + 1) * interval_eval))
            print("total time: {:.0f} minutes".format(np.sum(dict_result['t_train']) + np.sum(dict_result['t_pred']) + np.sum(dict_result['t_eval'])))

        print(dict_result)
        df_result = pd.DataFrame(dict_result)

        return df_result


    def do_params(self, params, df_train, df_vali, num_users=-1, num_items=-1, save_result_file=None):

        if params['recommender'] in ['RandomBase', 'random']:
            recommender = RandomBase(num_users=num_users, num_items=num_items,
                                     colname_user=self.colname_user, colname_item=self.colname_item,
                                     colname_outcome=self.colname_outcome, colname_prediction=self.colname_prediction)
        elif params['recommender'] in ['PopularBase', 'popular']:
            recommender = PopularBase(num_users=num_users, num_items=num_items,
                                      colname_user=self.colname_user, colname_item=self.colname_item,
                                      colname_outcome=self.colname_outcome, colname_prediction=self.colname_prediction)
        elif params['recommender'] in ['NeighborBase', 'neighbor', 'neighbor_base']:
            recommender = NeighborBase(num_users=num_users, num_items=num_items,
                                       colname_user=self.colname_user, colname_item=self.colname_item,
                                       colname_outcome=self.colname_outcome,
                                       colname_prediction=self.colname_prediction,
                                       measure_simil=params['measure_simil'],
                                       way_neighbor=params['way_neighbor'],
                                       num_neighbor=params['num_neighbor'],
                                       scale_similarity=params['scale_similarity'],
                                       normalize_similarity=params['normalize_similarity'])
        elif params['recommender'] == 'LMF':
            recommender = LMF(num_users=num_users, num_items=num_items,
                              colname_user=self.colname_user, colname_item=self.colname_item,
                              colname_outcome=self.colname_outcome, colname_prediction=self.colname_prediction,
                              dim_factor=params['dim_factor'], with_bias=params['with_bias'],
                              learn_rate=params['learn_rate'],
                              sd_init=params['sd_init'] / np.sqrt(params['dim_factor']),
                              reg_factor=params['reg_factor'], reg_bias=params['reg_bias'],
                              metric=params['train_metric'], ratio_nega=params['ratio_nega'])
        elif params['recommender'] == 'ULMF':
            recommender = ULMF(num_users=num_users, num_items=num_items,
                               colname_user=self.colname_user, colname_item=self.colname_item,
                               colname_outcome=self.colname_outcome, colname_prediction=self.colname_prediction,
                               dim_factor=params['dim_factor'], with_bias=params['with_bias'],
                               learn_rate=params['learn_rate'],
                               sd_init=params['sd_init'] / np.sqrt(params['dim_factor']),
                               reg_factor=params['reg_factor'], reg_bias=params['reg_bias'],
                               metric=params['train_metric'], ratio_nega=params['ratio_nega'],
                               alpha=params['alpha'])
        elif params['recommender'] == 'DLMF':
            recommender = DLMF(num_users=num_users, num_items=num_items,
                               colname_user=self.colname_user, colname_item=self.colname_item,
                               colname_outcome=self.colname_outcome, colname_prediction=self.colname_prediction,
                               dim_factor=params['dim_factor'], with_bias=params['with_bias'],
                               sd_init=params['sd_init'] / np.sqrt(params['dim_factor']),
                               learn_rate=params['learn_rate'], with_IPS=params['with_IPS'],
                               reg_factor=params['reg_factor'], reg_bias=params['reg_bias'],
                               reg_factor_j=params['reg_factor_j'], reg_bias_j=params['reg_bias_j'],
                               metric=params['train_metric'],
                               capping_T=params['capping_T'], capping_C=params['capping_C'],
                               only_treated=params['only_treated'],
                               coeff_T = params['coeff_T'], coeff_C = params['coeff_C'])

        elif params['recommender'] == 'CausE':
            recommender = CausE(num_users=num_users, num_items=num_items,
                               colname_user=self.colname_user, colname_item=self.colname_item,
                               colname_outcome=self.colname_outcome, colname_prediction=self.colname_prediction,
                               dim_factor=params['dim_factor'], with_bias=params['with_bias'],
                               learn_rate=params['learn_rate'],
                               sd_init=params['sd_init'] / np.sqrt(params['dim_factor']),
                               reg_factor=params['reg_factor'], reg_bias=params['reg_bias'],
                                reg_causal=params['reg_causal'],
                               metric=params['train_metric'])
        elif params['recommender'] == 'CausEProd':
            recommender = CausEProd(num_users=num_users, num_items=num_items,
                               colname_user=self.colname_user, colname_item=self.colname_item,
                               colname_outcome=self.colname_outcome, colname_prediction=self.colname_prediction,
                               dim_factor=params['dim_factor'], with_bias=params['with_bias'],
                               learn_rate=params['learn_rate'],
                               sd_init=params['sd_init'] / np.sqrt(params['dim_factor']),
                               reg_factor=params['reg_factor'], reg_bias=params['reg_bias'],
                                reg_causal=params['reg_causal'],
                               metric=params['train_metric'])
        else:
            pass


        # naive estimate of ITE (= individual treatment effect = causal effect of user-item pair)
        if params['naive']:
            empirical_mean_propensity = np.sum(df_train.loc[df_train.loc[:, 'idx_time']==0, self.colname_treatment].values)/(num_users * num_items)
            df_train.loc[:, self.colname_propensity] = empirical_mean_propensity
            print("empirical_mean_propensity: {}".format(empirical_mean_propensity))

        # Note that train data might be trimmed. Hence df_vali is used to calculate average IPS
        if params['mis_IPS_train'] != 0:  # train with misspecified propensity
            log_odds = np.log(
                df_train.loc[:, self.colname_propensity].values / (1 - df_train.loc[:, self.colname_propensity].values))
            log_odds_mean = np.nanmean(
                np.log(df_vali.loc[:, self.colname_propensity] / (1 - df_vali.loc[:, self.colname_propensity])))
            log_odds_mis = ((1 - params['mis_IPS_train']) * log_odds + params['mis_IPS_train'] * log_odds_mean)
            df_train.loc[:, self.colname_propensity] = 1 / (1 + np.exp(-log_odds_mis))

        if params['mis_IPS_eval'] != 0:  # evaluate with misspecified propensity
            log_odds = np.log(
                df_vali.loc[:, self.colname_propensity].values / (1 - df_vali.loc[:, self.colname_propensity].values))
            log_odds_mean = np.nanmean(log_odds)
            print(log_odds_mean)
            log_odds_mis = ((1 - params['mis_IPS_eval']) * log_odds + params['mis_IPS_eval'] * log_odds_mean)
            print(type(log_odds_mis))
            df_vali.loc[:, self.colname_propensity] = 1 / (1 + np.exp(-log_odds_mis))

        df_result = self.validate(recommender, df_train, df_vali, num_loop=params['num_loop'],
                                  interval_eval=params['interval_eval'], eval_metrics=params['eval_metrics'],
                                  eval_train_data=params['eval_train_data'], check_factors=params['check_factors'])
        return df_result


    def try_params(self, list_params, df_train, df_vali, num_users=-1, num_items=-1, save_result_file=None):

        if num_users < 0:
            num_users = np.max(df_train.loc[:, self.colname_user].values) + 1
        if num_items < 0:
            num_items = np.max(df_train.loc[:, self.colname_item].values) + 1

        print("num_users: {}".format(num_users))
        print("num_items: {}".format(num_items))

        for params in list_params:
            params = self.fill_defaults(params)
            params = self.set_common_reg(params)
            params = self.set_common_cap(params)
            params = self.set_common_coeff(params)

            df_result = self.do_params(params, df_train.copy(deep=True), df_vali.copy(deep=True),
                           num_users, num_items, save_result_file)

            print(df_result)
            for k,v in params.items():
                df_result[k] = v

            if 'df_all' in locals():
                df_all = df_all.append(df_result)
            else:
                df_all = df_result

            if save_result_file is not None:
                df_all.to_csv(save_result_file)

        return df_all


    def gen_grid_params(self, params_search):
        list_params = []
        for k, vs in params_search.items():
            list_params_new = []
            if len(list_params) > 0:
                for v in vs:
                    for params in list_params:
                        params[k] = v
                        list_params_new.append(params.copy())
                list_params = list_params_new.copy()
            else:
                for v in vs:
                    params = {k: v}
                    list_params_new.append(params)
                list_params = list_params_new.copy()

        return list_params

    def set_search_params(self, cond_search, type_search):
        if type_search[:4] == 'grid':
            cond_params = cond_search.split('+')
            params_search = dict()
            for cond_param in cond_params:
                conds = cond_param.split(':')
                params_search[conds[0]] = conds[1:]

            list_params = self.gen_grid_params(params_search)

        else:
            cond_params = cond_search.split('+')
            length_search = len(cond_params[0].split(':')) - 1
            list_params = []
            for n in range(length_search):
                params_search = dict()
                for cond_param in cond_params:
                    conds = cond_param.split(':')
                    params_search[conds[0]] = conds[n + 1]
                list_params.append(params_search)

        return list_params

    def set_common_params(self, list_params, common_params):
        for n in np.arange(len(list_params)):
            for k, v in common_params.items():
                list_params[n][k] = v
        return list_params

    def set_common_cap(self, params):
        if params['capping_T'] < 0:
            params['capping_T'] = params['capping']
            print("capping_T is set to {}".format(params['capping']))
        if params['capping_C'] < 0:
            params['capping_C'] = params['capping']
            print("capping_C is set to {}".format(params['capping']))
        return params
    
    def set_common_coeff(self, params):
        if params['coeff_T'] < 0:
            params['coeff_T'] = params['coeff_common']
            print("coeff_T is set to {}".format(params['coeff_common']))
        if params['coeff_C'] < 0:
            params['coeff_C'] = params['coeff_common']
            print("coeff_C is set to {}".format(params['coeff_common']))
        return params
        
    def set_common_reg(self, params):
        if params['reg_bias'] < 0:
            params['reg_bias'] = params['reg_common']
            print("reg_bias is set to {}".format(params['reg_common']))
        if params['reg_factor'] < 0:
            params['reg_factor'] = params['reg_common']
            print("reg_factor is set to {}".format(params['reg_common']))
        if params['reg_bias_j'] < 0:
            params['reg_bias_j'] = params['reg_bias']
            print("reg_bias_j is set to {}".format(params['reg_bias']))
        if params['reg_factor_j'] < 0:
            params['reg_factor_j'] = params['reg_factor']
            print("reg_factor_j is set to {}".format(params['reg_factor']))
        if params['reg_causal'] < 0:
            params['reg_causal'] = params['reg_common']
            print("reg_causal is set to {}".format(params['reg_common']))

        return params