'''
Some codes are refered from https://github.com/kengz/SLM-Lab, thanks for their hard works
'''
import torch
import numpy as np

epsilon = 1e-8

def calc_GAEs(rewards, dones, hatvs, gamma, lam):
    total_duration = len(rewards)
    # hatvs is the V(s_t) including the last state
    assert total_duration + 1 == len(hatvs)
    gaes = torch.zeros_like(rewards)
    future_gae = torch.tensor(0.0, dtype=rewards.dtype)
    # dones is used to indicate the episode boundary
    not_dones = 1 - dones
    for t in reversed(range(total_duration)):
        delta = rewards[t] + gamma * hatvs[t+1] * not_dones[t] - hatvs[t]
        future_gae = delta + future_gae * gamma * lam * not_dones[t]
        gaes[t] = future_gae
    return gaes

def calc_nsteps(rewards, dones, next_hatvs, gamma):
    total_duration = len(rewards)
    # next_hatvs is the last state of the reward function
    rets = torch.zeros_like(rewards)
    future_ret = next_hatvs
    # dones is used to indicate the episode boundary
    not_dones = 1 - dones
    for t in reversed(range(total_duration)):
        future_ret = rewards[t] + future_ret * gamma * not_dones[t]
        rets[t] = future_ret
    return rets

def standardize(v):
    '''Method to standardize a rank-1 np array'''
    assert len(v) > 1, 'Cannot standardize vector of size 1'
    v_std = (v - v.mean()) / (v.std() + epsilon)
    return v_std

def entropy(x):
    '''
    calculate the entropy of the give probability x
    :param x:
    :return: H
    '''
    H = - x * torch.log(x + epsilon)
    H = H.sum(-1)
    return H
