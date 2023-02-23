# coding=utf-8
# @Time : 2023/2/23 下午3:34
# @File : beiyesian.py
from collections import OrderedDict

import numpy as np
import scipy.stats as stats

history = [[6, 4, 2], [6, 3, 1], [5, 5, 1], [6, 3, 2], [5, 4, 3], [6, 6, 1], [4, 2, 1], [6, 4, 1], [6, 4, 2], [6, 4, 3],
           [6, 6, 1], [4, 3, 3], [6, 2, 1], [5, 5, 4], [4, 2, 1], [4, 4, 2], [6, 4, 2], [1, 5, 4], [6, 3, 3], [3, 3, 1], ]
# Define the prior probability distribution
counter_dict = OrderedDict()
for a in range(1, 7):
    for b in range(1, 7):
        for c in range(1, 7):
            idx = ''.join(sorted([str(a), str(b), str(c)]))
            counter_dict[idx] = 0

for h in history:
    idx = ''.join([str(x) for x in sorted(h)])
    counter_dict[idx] = counter_dict[idx] + 1
prior = np.array([v / (len(history)) for v in counter_dict.values()])


# Define the likelihood function
def likelihood(data, theta):
    return 1


# Define the posterior probability distribution
def posterior(data, prior, likelihood):
    unnorm_post = prior * likelihood(data, np.arange(1, 1 + len(prior)))
    return unnorm_post / np.sum(unnorm_post)


post = posterior(history, prior, likelihood)

# Calculate the mean and standard deviation of the posterior distribution
post_mean = np.dot(post, np.arange(1, 1 + len(prior)))
post_std = np.sqrt(np.dot(post, np.square(np.arange(1, 1 + len(prior)))) - np.square(post_mean))

# Predict the outcome of the next roll
pred_dist = stats.norm(post_mean, post_std)
pred_prob = [pred_dist.pdf(i) for i in range(1, 1 + len(prior))]
pred_prob /= np.sum(pred_prob)

max_idx = np.argmax(pred_prob)
print(list(counter_dict.keys())[max_idx])
s, b = np.split(pred_prob, 2)
print(np.sum(s))
print(np.sum(b))
print("small" if np.sum(s) > np.sum(b) else "big")
