from __future__ import division
import numpy as np
import pandas as pd
import argparse
import os
import ast
import matplotlib.pyplot as plt
import pprint

"""
Given a directory of results for grasping static object with planned motion,
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
    df_attempted = df.loc[df['grasp_attempted']]

    stats['num_successes'] = len(df_success)
    stats['num_trials'] = len(df)
    stats['num_attempted'] = len(df_attempted)
    stats['raw_success_rate'] = stats['num_successes'] / stats['num_trials']
    stats['attemped_success_rate'] = stats['num_successes'] / stats['num_attempted']
    return stats


if __name__ == '__main__':
    args = get_args()

    csv_names = os.listdir(args.result_dir)
    for n in csv_names:
        result_file_path = os.path.join(args.result_dir, n)
        df = pd.read_csv(result_file_path, index_col=0)
        stats = evaluate_results(df)

        print('')
        print(n)
        print("Statistics:")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(stats)