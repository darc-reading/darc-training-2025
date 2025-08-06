import numpy as np
import scipy.stats as stats

def nr_offspring(wgt):
    """ nr_kids = nr_offspring(wgt)
    """
    s_ens = wgt.shape
    K = s_ens[0]
    rem_Kw = np.mod(K * wgt, 1)
    floor_Kw = K * wgt - rem_Kw
    rem_K = np.sum(rem_Kw)
    rem_w = rem_Kw / rem_K
    mltn = np.random.multinomial(rem_K, rem_w.reshape((K,))).reshape((K, 1))
    nr_kids = floor_Kw + mltn
    nr_kids = nr_kids.astype(int)
    return nr_kids

def prediction(ens, wgt, std_dyn):
    """ ens = prediction(ens, wgt, std_dyn)
    """
    s_ens = ens.shape
    new_ens = np.zeros(s_ens)
    K = s_ens[0]
    nr_kids = nr_offspring(wgt)
    cnt = 0
    for k in range(0, K):
        for l in range(0, nr_kids[k, 0]):
            new_ens[cnt, 0] = np.mod((2 * ens[k, 0] + std_dyn * np.random.normal()), 1)
            cnt = cnt + 1

    return new_ens
            
def update(y, ens, std_obs):
    """ wgt = update(y, ens, std_obs)
    """
    s_ens = ens.shape
    wgt = np.zeros(s_ens)
    K = s_ens[0]
    for k in range(0, K):
        wgt[k, 0] = stats.norm.pdf(y - np.cos(2 * np.pi * ens[k, 0]), loc=0, scale=std_obs)
    wgt = wgt / np.sum(wgt)
    return wgt
