import numpy as np
import pandas
import argparse
import os
import ast
import matplotlib.pyplot as plt

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


def get_args():
    parser = argparse.ArgumentParser(description='Analyse Results for Dynamic Grasping Experiment')

    parser.add_argument('-r', '--results_dir', type=str, default='results',
                        help='Directory of results to be analysed. Ex: results')
    args = parser.parse_args()
    return args


def process_grasp_switch_stats(dfs):
    # data['grasp_switches_position_distances'] = data['grasp_switches_position_distances'].apply(
    #     lambda x: ast.literal_eval(x))

    stats_key = 'grasp_switches_position_distances'
    gs_dist_p = dfs[stats_key].apply(lambda x: ast.literal_eval(x))
    gs_dist_p = [dist for dist_list in gs_dist_p for dist in dist_list]
    plt.hist(gs_dist_p)
    plt.title(stats_key)
    plt.show()

    dfs.groupby(['object_name']).agg({'num_grasp_switches': 'mean'}).plot.bar()
    plt.show()


if __name__ == '__main__':
    args = get_args()
    result_filenames = os.listdir(args.results_dir)

    dfs = []
    for fname in result_filenames:
        dfs.append(pandas.read_csv(os.path.join(args.results_dir, fname)))
    data = pandas.concat(dfs)
    # data = data[data['object_name'] == 'cube']

    # process_grasp_switch_stats(data)

    avg_time_spent = np.mean(data['time_spent'])
    time_spent_stats = data.groupby(['success']).agg({'time_spent': 'mean'})
    success_rate = np.mean(data['success'])
    print('number of trials: {}, number of success: {}'.format(data.shape[0], np.sum(data['success'])))
    print('average time: {}'.format(avg_time_spent))
    print('success rate: {}'.format(success_rate))
    print('finished')

    result_key = 'success'
    print('Overall mean {}: \t'.format(np.mean(data[result_key])))
    success_rate_stats = data.groupby(['object_name']).agg({'success': 'mean'})
    print('success_rate_stats: \n {} \n'.format(success_rate_stats))

    import ipdb; ipdb.set_trace()
