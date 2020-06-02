import numpy as np

def get_train_dev_idx(N,dev_size=.2,random_state=0):

    if random_state is None:
        rand_idx = np.random.permutation(N)
    else:
        rs = np.random.RandomState(random_state)
        rand_idx = rs.permutation(N)

    if dev_size == 0:
        return rand_idx

    N_train = int(N * (1-dev_size))
    if N_train == N:
        print('Warning: dev_size too small!')
        N_train = N-1
    
    return rand_idx[:N_train], rand_idx[N_train:]


def get_kfolds_idx(N,k_folds=5,random_state=0):

    if random_state is None:
        rand_idx = np.random.permutation(N)
    else:
        rs = np.random.RandomState(random_state)
        rand_idx = rs.permutation(N)

    indeces = []
    splitted_arrays = np.array_split(rand_idx,k_folds)
    for i in range(1,k_folds+1):
        train_idx = np.hstack(splitted_arrays[:i-1] + splitted_arrays[i:])
        dev_idx = splitted_arrays[i-1]
        indeces.append((train_idx, dev_idx))

    return indeces