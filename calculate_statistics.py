from __future__ import division
import numpy as np
np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)

data = np.genfromtxt('results/cube.csv', delimiter=',', dtype=np.object)

time_spent = data[1:, 3].astype(np.float)
success = (data[1:, 0] == "True")
n_trials = 13
# time_spent = time_spent[-n_trials:]
# success = success[-n_trials:]

avg_time_spent = np.mean(time_spent[success])
success_rate = np.count_nonzero(success)/success.shape[0]
print("number of trials: {}, number of success: {}".format(success.shape[0], np.count_nonzero(success)))
print("average time: {}".format(avg_time_spent))
print("success rate: {}".format(success_rate))
print("finished")