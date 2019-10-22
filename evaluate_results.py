from __future__ import division
import numpy as np
import pandas as pd
import argparse
import os
import ast
import matplotlib.pyplot as plt
import pprint

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


def get_args():
    parser = argparse.ArgumentParser(description='Analyse results.')
    parser.add_argument('--grasp_dir', type=str, required=True,
                        help='File path to the result csv file')
    args = parser.parse_args()
    return args


def evaluate_results(df, success_rate_threshold):
    stats = {}
    df_success = df.loc[df['num_successes'] / df['num_trials'] >= success_rate_threshold]

    stats['num_success'] = len(df_success)
    stats['num_grasps'] = len(df)
    stats['success_rate'] = stats['num_success'] / stats['num_grasps']
    return stats


if __name__ == '__main__':
    args = get_args()

    object_names = os.listdir(args.grasp_dir)
    for n in object_names:
        result_file_path = os.path.join(args.grasp_dir, n, 'result.csv')
        df = pd.read_csv(result_file_path, index_col=0)
        stats = evaluate_results(df, success_rate_threshold=1.0)

        print('')
        print(n)
        print("Statistics:")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(stats)
