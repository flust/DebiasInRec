
from datetime import datetime
import numpy as np
import pandas as pd

import random
import os
random.seed(10)

from simulator import DataGenerator

import argparse

def setup_arg_parser():
    parser = argparse.ArgumentParser(
        prog='prepare_data.py',
        usage='prepare semi-synthetic data',
        description='',
        add_help=True)

    parser.add_argument('-d', '--dir_data', type=str, default='data/preprocessed/dunn_cat_mailer_10_10_1_1/',
                        help='path of data directory', required=False)
    parser.add_argument('-tlt', '--time_length_train', type=int,
                        help='length of training time', default=10,
                        required=True)
    parser.add_argument('-tlv', '--time_length_vali', type=int,
                        help='length of validation time', default=1,
                        required=False)
    parser.add_argument('-tle', '--time_length_eval', type=int,
                        help='length of evaluation time', default=10,
                        required=False)
    parser.add_argument('-rp', '--rate_prior', type=float,
                        help='rate of prior obtained from mean values', default=0.4,
                        required=False)
    parser.add_argument('-mas', '--mode_assignment', type=str,
                        help='mode of treatment assignment', default='original',
                        required=False)
    parser.add_argument('-sf', '--scale_factor', type=float,
                        help='scale factor for preference', default=2.0,
                        required=False)
    parser.add_argument('-nr', '--num_rec', type=int,
                        help='expected number of recommendation', default=210,
                        required=False)
    parser.add_argument('-cap', '--capping', type=float,
                        help='capping', default=0.000001,
                        required=False)
    parser.add_argument('-nc', '--num_CPU', type=int, default=1,
                        help='number of CPU used',
                        required=False)
    parser.add_argument('-ssr', '--set_seed_random', type=int, default=1,
                        help='set seed for randomness',
                        required=False)
    parser.add_argument('-trt', '--trim_train_data', action='store_true',
                        help='remove unpurchased for train data of idx_time > 0',
                        required=False)

    args = parser.parse_args()
    return args

def prepare_generator(path_empirical_data, capping, mode, scale_factor, num_rec, df_train=None):
    data_generator.load_data(path_empirical_data=path_empirical_data)
    data_generator.assign_propensity(capping=capping, mode=mode, scale_factor=scale_factor, num_rec=num_rec,
                                     df_train=df_train)

def generate(time_length, trim_data, with_additional_info=False):
    for n in np.arange(time_length):
        data_generator.assign_treatment()
        data_generator.assign_outcome()
        data_generator.assign_effect()
        if n == 0:
            df = data_generator.get_observation(with_additional_info=with_additional_info)
            df.loc[:, 'idx_time'] = n
        else:
            df_add = data_generator.get_observation(with_additional_info=with_additional_info)
            if trim_data:
                df_add = df_add.loc[df_add.loc[:, 'outcome'] + df_add.loc[:, 'treated'] + np.abs(df_add.loc[:, 'causal_effect']) > 0, :]
            df_add.loc[:, 'idx_time'] = n
            df = pd.concat([df, df_add])
    return df


if __name__ == '__main__':
    args = setup_arg_parser()
    if args.dir_data[-1] is not '/':
        args.dir_data = args.dir_data + '/'

    os.environ['OMP_NUM_THREADS'] = str(args.num_CPU)
    
    # start
    print('dir_data is {}.'.format(args.dir_data))

    if args.mode_assignment == 'original':
        dir_data_prepared = args.dir_data + args.mode_assignment + '_rp' + format(args.rate_prior, '.2f') + '/'
    else:
        dir_data_prepared = args.dir_data + args.mode_assignment + '_rp' + format(args.rate_prior, '.2f') + \
                            '_sf' + format(args.scale_factor, '.2f') + '_nr' + str(args.num_rec) +'/'

    # print(os.getcwd())
    # print(dir_data_prepared)
    # print(os.path.exists(dir_data_prepared))
    if not os.path.exists(dir_data_prepared):
        os.mkdir(dir_data_prepared)

    print('dir_data_prepared is {}.'.format(dir_data_prepared))
    print('Start prepare data.')
    t_init = datetime.now()
    data_generator = DataGenerator(rate_prior=args.rate_prior)
    np.random.seed(seed=args.set_seed_random)
    capping = args.capping
    print('mode_assignment is {}.'.format(args.mode_assignment))

    if 'kNN' in args.mode_assignment or 'BPR' in args.mode_assignment:
        df_original = pd.read_csv(args.dir_data + 'original' + '_rp' + format(args.rate_prior, '.2f') + '/data_train.csv')
        df_original = df_original.loc[df_original.loc[:, 'idx_time'] < 10, :]
        prepare_generator(path_empirical_data=args.dir_data + 'cnt_logs.csv',
                          capping=capping, mode=args.mode_assignment, df_train=df_original,
                          scale_factor=args.scale_factor, num_rec=args.num_rec)
    else:
        prepare_generator(path_empirical_data=args.dir_data + 'cnt_logs.csv',
                            capping=capping, mode=args.mode_assignment,
                           scale_factor=args.scale_factor, num_rec=args.num_rec)

    df_train = generate(time_length=args.time_length_train, trim_data=args.trim_train_data, with_additional_info=False)
    df_train.to_csv(dir_data_prepared + 'data_train.csv', index=False)

    df_vali = generate(time_length=args.time_length_vali, trim_data=False, with_additional_info=True)
    df_vali.to_csv(dir_data_prepared + 'data_vali.csv', index=False)

    df_test = generate(time_length=args.time_length_eval, trim_data=False, with_additional_info=False)
    df_test.to_csv(dir_data_prepared + 'data_test.csv', index=False)

    print('Data prepared.')
    print('Max propensity: {}'.format(np.max(df_vali.loc[:, 'propensity'])))
    print('Min propensity: {}'.format(np.min(df_vali.loc[:, 'propensity'])))
    print('Average propensity: {}'.format(np.mean(df_vali.loc[:, 'propensity'])))
    print('Average number of recommendations: {}'.format(np.mean(df_vali.loc[:, 'treated'])*data_generator.num_items))
    print('Ratio of positive outcomes: {}'.format(np.mean(df_vali.loc[:, 'outcome'])))
    print('Ratio of positive treatment effect: {}'.format(np.mean(df_vali.loc[:, 'causal_effect'] > 0)))
    print('Ratio of negative treatment effect: {}'.format(np.mean(df_vali.loc[:, 'causal_effect'] < 0)))
    print('Average treatment effect: {}'.format(np.mean(df_vali.loc[:, 'causal_effect'])))
    print('Expected average treatment effect: {}'.format(np.mean(df_vali.loc[:, 'causal_effect_expectation'])))

    save_result_dir = dir_data_prepared + 'result/'
    if not os.path.exists(save_result_dir):
        os.mkdir(save_result_dir)

    t_end = datetime.now()
    t_diff = t_end - t_init

    hours = (t_diff.days * 24) + (t_diff.seconds / 3600)

    print('Completed in {:.2f} hours.'.format(hours))


