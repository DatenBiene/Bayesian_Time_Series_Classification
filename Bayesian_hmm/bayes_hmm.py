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
    
    def simulate_transition(self, dirich):
        """ """
        n_states = dirich.shape[0]
        mat = np.zeros((n_states, n_states))
        for k in range(n_states):
            mat[k, :] = np.random.dirichlet(dirich[k, :])
        return mat
    
    def dirichlet_expectation(self, mat):
        row_sums = mat.sum(axis=1)
        return mat / row_sums[:, np.newaxis]
    
    def get_forward(self, y, prior_params, transition_matrix, initial_distribution=None):
        """ Only to get the normalizing constant for the backward step """
        c = []
        n_states = transition_matrix.shape[0]
        T = len(y)
        alpha = np.zeros((T, n_states))
        o = self.emission_proba_viterbi(y, transition_matrix, prior_params)
        
        if initial_distribution is None:
            initial_distribution = np.array([1./n_states] * n_states)
        alpha[0, :] = initial_distribution * o[:, 0]
        c0 = np.sum(alpha[0, :]) + 1e-6
        alpha[0, :] = alpha[0, :] / c0
        c.append(c0)
       
        for t in range(1, T):
            for j in range(n_states):
                alpha[t, j] = alpha[t - 1].dot(transition_matrix[:, j]) * o[j, t]
            ct = np.sum(alpha[t, :]) + 1e-6
            alpha[t, :] = alpha[t, :] / ct
            c.append(ct)
        
        self.forward = alpha
        self.c = c
        return None
    


    def get_backward(self, y, prior_params, transition_matrix, c):
        """ comments """
        n_states = transition_matrix.shape[0]
        T = len(y)
        backward = np.zeros((T, n_states))
        o = self.emission_proba_viterbi(y, transition_matrix, prior_params)
        # setting beta(T) = 1
        backward[T - 1] = np.ones((n_states)) /c[0]
        # Loop in backward way from T-1 to
        for t in range(T - 2, -1, -1):
            for j in range(n_states):
                backward[t, j] = (backward[t + 1] * o[:, t + 1]).dot(transition_matrix[j, :])
            ##### ---- experimental ------###### 
            #scaling to avoid underflow
            backward[t, :] = backward[t, :] / c[t]
        return backward
    
    def simulate_path(self, y, prior_params, transition_matrix):
        """ comments """
        self.get_forward(y, prior_params, transition_matrix)
        c = self.c
        backward = self.get_backward(y, prior_params, transition_matrix, c)
        T = len(y)
        path = []
        n_states = transition_matrix.shape[0]
        states = [s for s in range(n_states)]
        probas = [(1./n_states) * self.normal_pdf(y[0], prior_params[s][0], prior_params[s][1]) * backward[0, s] for s in states]
        #pas sur 
        probas = [min(1e5, v) for v in probas]
        probas = [max(1e-7, v) for v in probas]

        normalized_probas = np.array(probas) / np.sum(probas)
        x1 = np.random.choice(states, p=normalized_probas)
        path.append(x1)
        for t in range(1, T):
            probas = [transition_matrix[path[t-1], s] * self.normal_pdf(y[t], prior_params[s][0], prior_params[s][1]) * backward[t, s] for s in states]
            probas = [max(1e-7, v) for v in probas]
            probas = [min(1e5, v) for v in probas]
            normalized_probas = np.array(probas) / np.sum(probas)
            xt = np.random.choice(states, p=normalized_probas)
            path.append(xt)
        return path
    

    

    
    def fit(self, Y):
        """ comments """
        prior_transition = self.prior_transitions
        prior_params = self.prior_params
        n_iter_gibbs = self.n_iter_gibbs
        n = len(Y)
        mu_prior, n0_prior, alpha, beta = prior_params[0][0], prior_params[0][1], prior_params[0][2], prior_params[0][3]
        mu_posteriors = {}
        sigma_posteriors = {}
        states = np.arange(prior_transition.shape[0])
        for s in states:
            mu_posteriors[s] = []
            sigma_posteriors[s] = []
        params_step = prior_params.copy() 
        transition_matrix = self.simulate_transition(prior_transition)
        estimator_params = prior_params.copy()
        for k in range(n_iter_gibbs):
            hidden_seq_sim = self.simulate_path(Y, params_step, transition_matrix)
            prior_transition = self.update_prior_dirichlet(prior_transition, hidden_seq_sim)
            transition_matrix = self.simulate_transition(prior_transition)
            for s in states:
                
                y_s = Y[np.where(np.array(hidden_seq_sim) == s)]
                n_s = len(y_s)
                if n_s == 0:
                    continue
                y_bar = np.mean(y_s)
                ssy = np.sum(np.square(np.array(y_s) - y_bar))
                sigma_step = np.random.gamma(alpha + n_s/2, 1/(beta + 0.5*ssy + (n_s*n0_prior/(2*(n_s+n0_prior))) * (y_bar - mu_prior)**2))
                mu_step = np.random.normal((n_s/(n_s + n0_prior)) * y_bar + (n0_prior/(n_s + n0_prior)) * mu_prior, np.sqrt((n_s+n0_prior)/sigma_step))
                mu_posteriors[s].append(mu_step)
                sigma_posteriors[s].append(sigma_step)
                params_step[s][0], params_step[s][1] = mu_step, sigma_step  
                
            never_visited = set(np.arange(prior_transition.shape[0])) - set(params_step.keys())
            for s in never_visited:
                #putting mean to 0 and variance to 1. Arbitrary could find something smarter
                params_step[s][0], params_step[s][1] = 0, 1
        
        self.cache_mu_posteriors = mu_posteriors
        self.cache_sigma_posteriors = sigma_posteriors
        
        burn = min(1000, int(0.1 * n_iter_gibbs))
        for s in states:
            estimator_mu = np.mean(mu_posteriors[s][burn:])
            estimator_sigma = np.mean(sigma_posteriors[s][burn:])
            estimator_params[s][0], estimator_params[s][1] = estimator_mu, estimator_sigma 
            
        estimator_P = self.dirichlet_expectation(prior_transition)
        self.posterior_P, self.posterior_params = estimator_P, estimator_params
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
