import numpy as np
import random 
import scipy


def generate_markov_seq(n_states, transition_matrix, len_seq, init_state=None):
    states = [k for k in range(n_states)]
    seq = []
    if init_state:
        x0 = init_state
    else:
        x0 = np.random.choice(states) #add initial probabilities
    x_prev = x0
    seq.append(x_prev)
    for i in range(len_seq):
        x_succ = np.where(np.random.multinomial(1, transition_matrix[x_prev, :], size=1) == 1)[1][0]
        seq.append(x_succ)
        x_prev = x_succ
    return seq

def generate_transtion_matrix(n_states):
    mat = []
    for k in range(n_states):
        row = np.random.random(n_states)
        row = row / np.sum(row)
        mat.append(list(row))
    return np.array(mat)

def generate_series(hidden_seq, params):
    T = len(hidden_seq)
    y = []
    for t in range(T):
        mu_step = params[hidden_seq[t]][0]
        sigma_step = params[hidden_seq[t]][1]
        y.append(np.random.normal(mu_step, sigma_step))
    return y


def generate_samples(n_sample, lengths_range, P, params, noise=0., init_state=None):
    Y = []
    for sample in range(n_sample):
        n_states = P.shape[0]
        T = np.random.randint(lengths_range[0], lengths_range[1])
        hidden_seq = generate_markov_seq(n_states, P, T, init_state) #hidden states sequence 
        y = generate_series(hidden_seq, params) #time series following HMM model with hidden states and params
        y = np.array(y) + np.random.random(len(y)) * noise #adding noise to series
        Y.append(y)
    return Y
