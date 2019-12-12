import numpy as np
import random 
import scipy
import matplotlib.pyplot as plt
from random import shuffle

class bayesian_hmm:
    def __init__(self, prior_params, prior_transitions, n_iter_gibbs, class_label=None):
        self.prior_params = prior_params
        self.prior_transitions = prior_transitions
        self.n_iter_gibbs = n_iter_gibbs
        self.class_label = class_label
        self.cache_estimator = {}
            
    def normal_pdf(self, x, mu, sigma):
        return np.exp(-(x - mu)**2/(2*sigma))/(np.sqrt(2*np.pi)*sigma)
    
    def update_prior_dirichlet(self, prior, hidden_seq_obs):
        states = list(set(hidden_seq_obs))
        states.sort()
        n_states = len(states)
        T = len(hidden_seq_obs)
        posterior = prior.copy()
        for h in range(1, T):
            current = hidden_seq_obs[h]
            prev = hidden_seq_obs[h-1]
            posterior[prev, current] += 1
        return posterior

    def dirichlet_expectation(self, mat):
        row_sums = mat.sum(axis=1)
        return mat / row_sums[:, np.newaxis]
    
    def get_forward(self, y, prior_params, prior_transitions, initial_distribution=None):
        """ Only to get the normalizing constant for the backward step """
        c = []
        n_states = prior_transitions.shape[0]
        T = len(y)
        alpha = np.zeros((T, n_states))
        o = self.emission_proba_viterbi(y, prior_transitions, prior_params)
        
        if initial_distribution is None:
            initial_distribution = np.array([1./n_states] * n_states)
        alpha[0, :] = initial_distribution * o[:, 0]
        c0 = np.sum(alpha[0, :])
        alpha[0, :] = alpha[0, :] / c0
        c.append(c0)
       
        for t in range(1, T):
            for j in range(n_states):
                alpha[t, j] = alpha[t - 1].dot(prior_transitions[:, j]) * o[j, t]
            ct = np.sum(alpha[t, :])
            alpha[t, :] = alpha[t, :] / ct
            c.append(ct)
        
        self.forward = alpha
        self.c = c
        return None
    


    def get_backward(self, y, prior_params, prior_transitions, c):
        """ comments """
        n_states = prior_transitions.shape[0]
        T = len(y)
        backward = np.zeros((T, n_states))
        o = self.emission_proba_viterbi(y, prior_transitions, prior_params)
        # setting beta(T) = 1
        backward[T - 1] = np.ones((n_states)) /c[0]
        # Loop in backward way from T-1 to
        for t in range(T - 2, -1, -1):
            for j in range(n_states):
                backward[t, j] = (backward[t + 1] * o[:, t + 1]).dot(prior_transitions[j, :])
            ##### ---- experimental ------###### 
            #scaling to avoid underflow
            backward[t, :] = backward[t, :] / c[t]
        return backward
    
    def simulate_path(self, y, prior_params, prior_transitions):
        """ comments """
        self.get_forward(y, prior_params, prior_transitions)
        c = self.c
        backward = self.get_backward(y, prior_params, prior_transitions, c)
        T = len(y)
        path = []
        n_states = prior_transitions.shape[0]
        states = [s for s in range(n_states)]
        probas = [1./n_states * self.normal_pdf(y[0], prior_params[s][0], prior_params[s][1]) * backward[0, s] for s in states]
        normalized_probas = np.array(probas) / np.sum(probas)
        x1 = np.random.choice(states, p=normalized_probas)
        path.append(x1)
        for t in range(1, T):
            probas = [prior_transitions[path[t-1], s] * self.normal_pdf(y[t], prior_params[s][0], prior_params[s][1]) * backward[t, s] for s in states]
            normalized_probas = np.array(probas) / np.sum(probas)
            xt = np.random.choice(states, p=normalized_probas)
            path.append(xt)
        return path
    
    def update_prior_gaussian(self, prior, y, n_iter, state):
        #gibbs sampling
        mu_prior, n0_prior, alpha, beta = prior[0], prior[1], prior[2], prior[3]
        y_bar = np.mean(y)
        ssy = np.sum(np.square(np.array(y) - y_bar))
        n = len(y)
        mu_posteriors = [mu_prior]
        sigma_posteriors = []
        for k in range(n_iter):
            sigma_step = np.random.gamma(alpha + n/2, 1/(beta + 0.5*ssy + (n*n0_prior/(2*(n+n0_prior))) * (y_bar - mu_prior)**2))
            mu_step = np.random.normal((n/(n + n0_prior)) * y_bar + (n0_prior/(n + n0_prior)) * mu_prior, np.sqrt((n+n0_prior)/sigma_step))
            mu_posteriors.append(mu_step)
            sigma_posteriors.append(sigma_step)
        
        #me, se = dict(), dict()
        #me[state]['mu'] = mu_posteriors
        #se[state]['sigma'] = sigma_posteriors
        #self.cache_estimator.update(me)
        #self.cache_estimator.update(se)
        burn = min(1000, int(0.1 * n_iter))
        estimator_mu = np.mean(mu_posteriors[burn:])
        estimator_sigma = np.mean(sigma_posteriors[burn:])
        return estimator_mu, np.sqrt(1/estimator_sigma)
    
    def fit_one_obs(self, y, labels, prior_params, prior_transitions, n_iter_gibbs):
        a = self.update_prior_dirichlet(prior_transitions, labels)
        P_hat = self.dirichlet_expectation(a)
        states = set(labels)
        params_hat = {}
        for s in states:
            y_s = y[np.where(np.array(labels) == s)]
            params_hat[s] = self.update_prior_gaussian(prior_params[s], y_s, n_iter_gibbs, s)
            
        #Dealing with the case where labels does not contains all possible states 
        # i.e. one state is never visited
        never_visited = set(np.arange(P_hat.shape[0])) - set(params_hat.keys())
        for s in never_visited:
            #putting mean to 0 and variance to 1. Arbitrary could find something smarter
            params_hat[s] = 0, 1
        return P_hat, params_hat
    
    def fit(self, Y):
        
            
        prior_transition = self.prior_transitions
        prior_params = self.prior_params
        n_iter_gibbs = self.n_iter_gibbs
        for obs in Y:
            #y_obs, hidden_seq_obs = obs[0], obs[1]
            y_obs = obs
            hidden_seq_sim = self.simulate_path(y_obs, prior_params, prior_transition)
            fit = self.fit_one_obs(y_obs, hidden_seq_sim, prior_params, prior_transition, n_iter_gibbs)
            for s in prior_params:
                try:
                    prior_params[s][0], prior_params[s][1] = fit[1][s][0], fit[1][s][1]
                except:
                    print("error")
                    return prior_params, fit[1]
            prior_transition = fit[0]
        self.posterior_P, self.posterior_params = prior_transition, prior_params
        print("Model fitted")
    
    def emission_proba_viterbi(self, y, P_hat, params_hat):
        n_states = P_hat.shape[0]
        B = np.zeros((n_states, len(y)))
        for s in range(B.shape[0]):
            mu_s = params_hat[s][0]
            sigma_s = params_hat[s][1]
            for t in range(B.shape[1]):
                o_t = y[t]
                B[s, t] = self.normal_pdf(o_t, mu_s, sigma_s)
        return B
    
    def viterbi(self, y, A, B, Pi=None):
        """Slightly modified version of viterbi algorithm from wikipedia
        to suits our needs."""
        # Cardinality of the state space
        K = A.shape[0]
        # Initialize the priors with default (uniform dist) if not given 
        Pi = Pi if Pi is not None else np.full(K, 1 / K)
        T = len(y)
        T1 = np.empty((K, T), 'd')
        T2 = np.empty((K, T), 'B')

        # We take the log of the probabilities to avoid vanishing proba - underflow
        T1[:, 0] = np.log(Pi) + np.log(B[:, 0])
        T2[:, 0] = 0
        # Iterate throught the observations updating the tracking tables
        for i in range(1, T):
            T1[:, i] = np.max(T1[:, i - 1] + np.log(A.T) + np.log(B[np.newaxis, :, i].T), 1)
            #print(T1[:, i - 1] * A.T * B[np.newaxis, :, i].T)
            T2[:, i] = np.argmax(T1[:, i - 1] + np.log(A.T), 1)

        # Build the output, optimal model trajectory
        x = np.empty(T, 'B')
        x[-1] = np.argmax(T1[:, T - 1])
        for i in reversed(range(1, T)):
            x[i - 1] = T2[x[i], i]

        return x, T1[0] 
    
    def likelihood(self, new_series):
        try: 
            self.posterior_P 
        except:
            raise Exception("Need to call fit before likelihood")
            
        B = self.emission_proba_viterbi(new_series, self.posterior_P, self.posterior_params)
        _, probas = self.viterbi(new_series, self.posterior_P, B)
        return probas[-1]
    
    
    def monitor_convergence(self, state): 
        plt.figure(figsize=(7,7))
        plt.plot(self.cache_estimator[state]['mu'], label='mu')
        plt.title("Convergence of mean for state :"+str(state))
        plt.xlabel("Gibbs iterations")
        plt.legend()
        plt.show()
        plt.figure(figsize=(7,7))
        plt.plot(self.cache_estimator[state]['sigma'], label='sigma')
        plt.title("Convergence of variance for state :"+str(state))
        plt.xlabel("Gibbs iterations")
        plt.legend()
        plt.show()
