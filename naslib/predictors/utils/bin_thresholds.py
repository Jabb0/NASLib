"""
This file contains methods to discretize continuous features.
It is currently used by omni_seminas to create a discrete encoding
for zero-cost and LCE features.

In a real experiment, we would need to estimate the upper bounds for 
each bin during the search. To save time, we precomputed zero-cost
bins and then add the runtime of this precomputation later.
"""

def discretize(x, upper_bounds=None, one_hot=True):
    # return discretization based on upper_bounds
    # supports one_hot or categorical output
    assert upper_bounds is not None and len(upper_bounds) >= 1

    if one_hot:
        cat = len(upper_bounds) + 1
        discretized = [0 for _ in range(cat)]
        for i, ub in enumerate(upper_bounds):
            if x < ub:
                discretized[i] = 1
                return discretized
        discretized[-1] =  1
        return discretized
    else:
        for i, ub in enumerate(upper_bounds):
            if x < ub:
                return i
        return len(upper_bounds) + 1

def get_bins(zero_cost, train_size, ss_type, dataset):

    if ss_type == 'nasbench201' and dataset == 'cifar10' and zero_cost == 'jacov':
        # precomputation based on 100 jacov values (366 sec on a CPU)
        if train_size <= 10:
            bins = [-317.264]
        elif train_size <= 20:
            bins = [-460.405, -282.114]
        elif train_size <= 40:
            bins = [-702.848, -317.264, -280.275]
        elif train_size <= 80:
            bins = [-2017.64, -460.571, -317.621, -282.179, -279.084]
        else:
            # precompution based on 1000 jacov values (3660 sec on a CPU)
            bins = [-20893.873, -1179.832, -518.407, -373.523, -317.264, 
                    -284.944, -281.242, -279.503, -278.083]
    else:
        raise NotImplementedError('Currently no other zero-cost methods are supported')
    
    return bins

def get_lce_bins(train_info, key='TRAIN_LOSS_lc', max_bin_size=9):

    train_size = len(train_info)
    losses = sorted([i[key][-1] for i in train_info])
    n = min(max_bin_size, max(1, train_size//5))

    if n == 1:
        return [losses[train_size//2]]
    n_range = range(train_size//n, train_size, train_size//n)
    if len([i for i in n_range]) >= n:
        n_range = range(train_size//n, train_size - 1, train_size//n)

    return [losses[i] for i in n_range]