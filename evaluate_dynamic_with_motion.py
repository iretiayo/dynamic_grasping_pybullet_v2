from __future__ import division
import numpy as np
import pandas as pd
import argparse
import os
import ast
import matplotlib.pyplot as plt
import pprint

"""
Given a directory of results for grasping dynamic object with planned motion,
evaluate success rates.
"""

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


def get_args():
    parser = argparse.ArgumentParser(description='Analyse results.')
    parser.add_argument('--result_dir', type=str, required=True)
    args = parser.parse_args()
    return args


def evaluate_results(df):
    stats = {}
    df_success = df.loc[df['success']]

    stats['num_successes'] = len(df_success)
    stats['num_trials'] = len(df)
    stats['success_rate'] = stats['num_successes'] / stats['num_trials']
    stats['dynamic_grasping_time'] = df.mean().dynamic_grasping_time
    return stats


def get_overall_stats(stat_list):
    overall_stats = {}

    num_successes_list = []
    num_trials_list = []
    success_rate_list = []
    dynamic_grasping_time_list = []
    for stats in stat_list:
        num_successes_list.append(stats['num_successes'])
        num_trials_list.append(stats['num_trials'])
        success_rate_list.append(stats['success_rate'])
        dynamic_grasping_time_list.append(stats['dynamic_grasping_time'])

    overall_stats['num_successes'] = sum(num_successes_list)
    overall_stats['num_trials'] = sum(num_trials_list)
    overall_stats['success_rate'] = overall_stats['num_successes'] / overall_stats['num_trials']
    overall_stats['dynamic_grasping_time'] = np.average(dynamic_grasping_time_list)
    return overall_stats


if __name__ == '__main__':
    args = get_args()

    csv_names = os.listdir(args.result_dir)
    stat_list = []
    for n in csv_names:
        result_file_path = os.path.join(args.result_dir, n)
        df = pd.read_csv(result_file_path, index_col=0)
        stats = evaluate_results(df)
        stat_list.append(stats)

        print('')
        print(n)
        print("Statistics:")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(stats)

    overall_stats = get_overall_stats(stat_list)
    print('')
    print('Summary')
    print("Statistics:")
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(overall_stats)

