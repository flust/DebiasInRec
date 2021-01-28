
from datetime import datetime
import numpy as np
import pandas as pd

import random
import os
random.seed(10)

from experimenter import Experimenter

import argparse

def setup_arg_parser():
    parser = argparse.ArgumentParser(
        prog='param_search.py',
        usage='hyper parameter search',
        description='',
        add_help=True)

    parser.add_argument('-d', '--dir_data', type=str, default='data/preprocessed/dunn_cat_mailer_10_10_1_1/',
                        help='path of data directory', required=False)
    # conditions for data preparation (construct dir_data_prepared)
    parser.add_argument('-tlt', '--time_length_train', type=int,
                        help='length of training time', default=10,
                        required=False)
    parser.add_argument('-tle', '--time_length_eval', type=int,
                        help='length of evaluation time', default=10,
                        required=False)
    parser.add_argument('-rp', '--rate_prior', type=float,
                        help='rate of prior obtained from mean values', default=0.4,
                        required=True)
    parser.add_argument('-mas', '--mode_assignment', type=str,
                        help='mode of treatment assignment', default='original',
                        required=True)
    parser.add_argument('-sf', '--scale_factor', type=float,
                        help='scale factor for preference', default=2.0,
                        required=False)
    parser.add_argument('-nr', '--num_rec', type=int,
                        help='expected number of recommendation', default=210,
                        required=False)

    # conditions for the experiment
    parser.add_argument('-tm', '--type_model', type=str,
                        help='type of model',
                        required=True)
    parser.add_argument('-nc', '--num_CPU', type=int, default=1,
                        help='number of CPU used',
                        required=False)
    parser.add_argument('-cs', '--cond_search', type=str, default='dim_factor:100+reg_common:0.0001:0.001:0.01+learn_rate:0.001',
                        help='condition of search',
                        required=True)
    parser.add_argument('-ts', '--type_search', type=str, default='grid',
                        help='type of search',
                        required=False)
    parser.add_argument('-nu', '--num_users', type=int, default=-1,
                        help='number of users',
                        required=False)
    parser.add_argument('-ni', '--num_items', type=int, default=-1,
                        help='number of items',
                        required=False)
    parser.add_argument('-p', '--phase', type=str, default='validation',
                        help='phase from validation, test',
                        required=False)
    parser.add_argument('-ce', '--check_estimator', type=bool, default=False,
                        help='check estimator',
                        required=False)
    parser.add_argument('-sm', '--save_model', type=bool, default=False,
                        help='save model or not',
                        required=False)
    parser.add_argument('-se', '--save_each', type=bool, default=False,
                        help='save each result or not',
                        required=False)
    parser.add_argument('-ssr', '--set_seed_random', type=int, default=1,
                        help='set seed for randomness',
                        required=False)
    parser.add_argument('-ne', '--name_experiment', type=str, default='exp',
                        help='abbreviated name to express the experiment',
                        required=False)
 
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = setup_arg_parser()
    args.dir_data = 'data/preprocessed/' + args.dir_data
    if args.dir_data[-1] is not '/':
        args.dir_data = args.dir_data + '/'

    os.environ['OMP_NUM_THREADS'] = str(args.num_CPU)
    
    if args.mode_assignment == 'original':
        dir_data_prepared = args.dir_data + args.mode_assignment + '_rp' + format(args.rate_prior, '.2f') + '/'
    elif args.mode_assignment in ['kNN', 'oracle']:
        dir_data_prepared = args.dir_data + args.mode_assignment + '_rp' + format(args.rate_prior, '.2f') + \
                            '_sf' + format(args.scale_factor, '.2f') + '_nr' + str(args.num_rec) + '/'
    else:
        dir_data_prepared = args.dir_data + args.mode_assignment + '_rp' + format(args.rate_prior, '.2f') + \
                            '_sf' + format(args.scale_factor, '.2f') + '_nr' + str(args.num_rec) +'/'

    if args.phase == 'validation':
        phase = 'valid_phase'
    else:
        phase = 'test_phase'

    # prepare data
    if phase == 'test_phase':
        df_vali = pd.read_csv(dir_data_prepared + 'data_test.csv')
        max_time = np.max(df_vali.loc[:, 'idx_time']) + 1
        if args.time_length_eval < max_time:
            df_vali = df_vali.loc[df_vali.loc[:, 'idx_time'] < args.time_length_eval, :]
    else:
        df_vali = pd.read_csv(dir_data_prepared + 'data_vali.csv')

    df_train = pd.read_csv(dir_data_prepared + 'data_train.csv')
    df_train = df_train.loc[df_train.loc[:, 'idx_time'] < args.time_length_train, :]

    # to reduce memory usage
    df_vali = df_vali.astype({'idx_user': 'uint32', 'idx_item': 'uint32', 'outcome': 'int8', 'idx_time': 'int8', 'causal_effect': 'int8'})
    df_train = df_train.astype({'idx_user': 'uint32', 'idx_item': 'uint32', 'outcome': 'int8', 'idx_time': 'int8', 'causal_effect': 'int8'})

    print('Data loaded.')
    # set common params
    common_params = dict()
    common_params['num_CPU'] = args.num_CPU
    common_params['save_model_dir'] = dir_data_prepared + args.type_model + '_' + datetime.now().strftime(
        '%Y%m%d_%H%M%S') +  '/'
    common_params['recommender'] = args.type_model

    # choose evaluation metrics
    if args.phase == 'test':
        common_params['eval_metrics'] = 'Prec_10-Prec_100-DCG_100000-AR-AUC' + '-CPrec_10-CPrec_100-CDCG_100000-CAR'
        if args.check_estimator:
            common_params['eval_metrics'] += \
                    '-CPrecIPSin0.0_10-CPrecIPSin0.0_100-CDCGIPSin0.0_100000-CARIPSin0.0' + \
                    '-CPrecIPSin0.001_10-CPrecIPSin0.001_100-CDCGIPSin0.001_100000-CARIPSin0.001' + \
                    '-CPrecIPSin0.003_10-CPrecIPSin0.003_100-CDCGIPSin0.003_100000-CARIPSin0.003' + \
                    '-CPrecIPSin0.01_10-CPrecIPSin0.01_100-CDCGIPSin0.01_100000-CARIPSin0.01' + \
                    '-CPrecIPSin0.03_10-CPrecIPSin0.03_100-CDCGIPSin0.03_100000-CARIPSin0.03' + \
                    '-CPrecIPSin0.1_10-CPrecIPSin0.1_100-CDCGIPSin0.1_100000-CARIPSin0.1' + \
                    '-CPrecIPSin0.3_10-CPrecIPSin0.3_100-CDCGIPSin0.3_100000-CARIPSin0.3'
    else:
        common_params['eval_metrics'] = 'Prec_10-Prec_100' + '-CPrec_10-CPrec_100-CDCG_100000-CAR'
        if args.check_estimator:
            common_params['eval_metrics'] += '-CPrecIPSin0.01_10-CPrecIPSin0.01_100-CDCGIPSin0.01_100000-CARIPSin0.01'

    # set up experimenter
    experimenter = Experimenter()
    list_params = experimenter.set_search_params(args.cond_search, args.type_search)
    list_params = experimenter.set_common_params(list_params, common_params)
    save_result_file = dir_data_prepared + "result/" + datetime.now().strftime(
        '%Y%m%d_%H%M%S') + "_" + args.type_model + "_" + args.name_experiment + '_tlt' + str(args.time_length_train) + ".csv"

    if phase == 'test_phase':
        save_result_file = save_result_file.replace('.csv', '_test.csv')

    print('save_result_file is {}'.format(save_result_file))
    save_result_dir = os.path.dirname(save_result_file)
    if not os.path.exists(save_result_dir):
        os.mkdir(save_result_dir)

    print('Start experiment.')
    t_init = datetime.now()
    df_result = experimenter.try_params(list_params, df_train, df_vali, args.num_users, args.num_items, save_result_file)
    t_end = datetime.now()
    t_diff = t_end - t_init

    hours = t_diff.days * 24 + t_diff.seconds/(60 * 60)

    df_result.to_csv(save_result_file)
    print('Completed in {:.2f} hours.'.format(hours))
