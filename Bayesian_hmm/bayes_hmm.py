import numpy as np
import random 
import scipy
import matplotlib.pyplot as plt
from random import shuffle


class bayesian_hmm:
    """ Hidden Markov Model object with bayesian estimation of the parameters (theta, P). The number of Hidden states
        is an implicit parameters and is not estimated. For now this class handles only continuous observations where each observation
        has a conditionnal Gaussian distribution given the hidden state.
        Note : under this assumption, the entire sequence of observation would typically follow a mixture of Gaussian distribution,
        therefore it's good idea to plot the distribution of the sequence before using this model. 
        Can be used for classification and for segmentation (to come) of time series. 
        ----------------
            Parameters :
            See github and reference article for details about paramters
            prior_params : dict state : list [prior mean, lambda, alpha, beta]
            prior_transitions : array of shape (n_state, n_state) corresponding to the dirichlet prior on the transition matrix
            of the hidden states
            n_iter_gibbs : number of iteration for the Gibbs sampling 
            class_label : Label of the class if model is used for classification.
            -----------
            
        """
    def __init__(self, prior_params, prior_transitions, n_iter_gibbs, class_label=None):
        self.prior_params = prior_params
        self.prior_transitions = prior_transitions
        self.n_iter_gibbs = n_iter_gibbs
        self.class_label = class_label
        self.cache_estimator = {}
        self.n_states = prior_transitions.shape[0]
            
    def normal_pdf(self, x, mu, sigma):
        """ PDF of a Gaussian N(mu, sigma)
        at x.
        """
        return np.exp(-(x - mu)**2/(2*sigma))/(np.sqrt(2*np.pi)*sigma)
    
    def update_prior_dirichlet(self, prior, hidden_seq_obs):
        """ Updates the prior dirichlet distribution of the transition matrix using a sequence of hidden states 
            -----------
            Parameters :
            prior : array of shape (n_state, n_state) corresponding to the dirichlet prior on the transition matrix
            of the hidden states
            hidden_seq_obs : list of length > 0 corresponding to a sequence of hidden states
            -----------
            Returns : 
            Posterior : array of shape (n_state, n_state) corresponding to the posterior dirichlet on the transition matrix
            of the hidden states given the sequence
        """
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
        """ Simulates a transition matrix based on dirich distribution
            -----------
            Parameters :
            dirich : array of shape (n_state, n_state) corresponding to the dirichlet distribution on the transition matrix
            of the hidden states
            -----------
            Returns : 
            mat : array of shape (n_state, n_state) corresponding to a realisation of the dirich distribution
        """
        n_states = dirich.shape[0]
        mat = np.zeros((n_states, n_states))
        for k in range(n_states):
            mat[k, :] = np.random.dirichlet(dirich[k, :])
        return mat
    
    def dirichlet_expectation(self, mat):
        """ Compute the expectation of a dirichlet distribution (actually it's several dirichlet distributions, one per state)
            -----------
            Parameters :
            mat : array of shape (n_state, n_state), each row is a dirichlet parameter
            -----------
            Returns : 
            Array of shape (n_state, n_state) : The i-th row of is the expectation of Dirchlet(i-th row of initial mat).
        """
        row_sums = mat.sum(axis=1)
        return mat / row_sums[:, np.newaxis]

    

    
    def get_backward(self, y, prior_params, transition_matrix):
        """ Compute the forward probablities of the HMM using the backward algorithm
            -----------
            Parameters :
            y : array - like object of length T, sequence of observation (time series)
            prior_params : dict state : list [prior mean, lambda, alpha, beta] 
            transition_matrix : array of shape (n_state, n_state) corresponding to the transition matrix of the hidden sequence
            -----------
            Returns : 
            backward : array of shape (T, n_state)
        """
        n_states = transition_matrix.shape[0]
        T = len(y)
        backward = np.zeros((T, n_states))
        o = self.emission_proba_viterbi(y, transition_matrix, prior_params)
        o[np.where(o == 0)] = 1e-7
        # setting beta(T) = 1
        backward[T - 1] = np.ones((n_states)) / n_states
        # Loop in backward way from T-1 to
        for t in range(T - 2, -1, -1):
            for j in range(n_states):
                backward[t, j] = (backward[t + 1] * o[:, t + 1]).dot(transition_matrix[j, :])
            ##### ---- experimental ------###### 
            #scaling to avoid underflow
            if backward[t, :].sum() == 0:
                print(backward[t, :], o[:, t+1])
            backward[t, :] = backward[t, :] / backward[t, :].sum()
        return backward
    
    
    def simulate_path(self, y, prior_params, transition_matrix):
        """ Simulates a path through the hidden states given the data and the parameters.
            See reference article for details about distributions from which path is sampled.
            -----------
            Parameters :
            y : array - like object of length T, sequence of observation (time series)
            prior_params : dict state : list [prior mean, lambda, alpha, beta] 
            transition_matrix : array of shape (n_state, n_state) corresponding to the transition matrix of the hidden sequence 
            -----------
            Returns : 
            list of length T corresponding to a sequence of hidden states.
        """
        
        backward = self.get_backward(y, prior_params, transition_matrix)
        T = len(y)
        path = []
        n_states = transition_matrix.shape[0]
        states = [s for s in range(n_states)]
        probas = [(1./n_states) * self.normal_pdf(y[0], prior_params[s][0], prior_params[s][1]) * backward[0, s] for s in states]
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
        """ Fit the HMM to the Data. All parameters are estimated using Gibbs Sampling and Block Gibbs sampling.
            See reference article for technical details.
            -----------
            Parameters :
            Y : array - like object of length T, sequence of observation (time series)
            -----------
            Returns : 
            None
        """
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
                if n_s < 1:
                    continue
                y_bar = np.mean(y_s)
                ssy = np.sum(np.square(np.array(y_s) - y_bar))
                sy = np.sum(y_s)
                scale = 1./( beta + 0.5*(ssy + (n0_prior*n_s*(y_bar - mu_prior)**2)/(n0_prior + n_s) ) )
                sigma = np.random.gamma(alpha + (n_s/2), scale)
                sigma_step = 1./ sigma
                mu_step = np.random.normal( (n0_prior*mu_prior + n_s * y_bar)/(n_s + n0_prior), np.sqrt(sigma_step/(n_s + n0_prior)) )
                
                mu_posteriors[s].append(mu_step)
                sigma_posteriors[s].append(sigma_step)
                params_step[s][0], params_step[s][1] = mu_step, sigma_step  
                
            never_visited = set(np.arange(prior_transition.shape[0])) - set(params_step.keys())
            for s in never_visited:
                #putting mean to 0 and variance to 1. Arbitrary could find something smarter
                params_step[s][0], params_step[s][1] = 0, 1
        
        self.cache_mu_posteriors = mu_posteriors
        self.cache_sigma_posteriors = sigma_posteriors
        #self.cache_estimator = {'mu': mu_posteriors, 'sigma':sigma_posteriors}
        
        burn = min(1000, int(0.1 * n_iter_gibbs))
        for s in states:
            estimator_mu = np.mean(mu_posteriors[s][burn:])
            estimator_sigma = np.mean(sigma_posteriors[s][burn:])
            estimator_params[s][0], estimator_params[s][1] = estimator_mu, estimator_sigma 
            
        estimator_P = self.dirichlet_expectation(prior_transition)
        self.posterior_P, self.posterior_params = estimator_P, estimator_params
        print("Model fitted")

    
    def emission_proba_viterbi(self, y, P_hat, params_hat):
        """ Computes the emission probabilities of each observation in the sequence given the sequence and the parameters.
            Since the observations are continuous, we compute it several times each time the parameters change.
            -----------
            Parameters :
            y : array - like object of length T, sequence of observation (time series)
            params_hat : dict state : list [prior mean, lambda, alpha, beta] 
            P_hat : array of shape (n_state, n_state) corresponding to the transition matrix of the hidden sequence 
            -----------
            Returns : 
            B : array of shape (n_state, T). B[i, j] is the probability of observing y[j] given that we are under the hidden state i. 
        """
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
        """ Slightly modified version of viterbi algorithm from wikipedia
            to suits our needs. https://en.wikipedia.org/wiki/Viterbi_algorithm
            In particular we compute log likelihoods instead of probabilities to avoid underflow.
            -----------
            Parameters :
            y : array - like object of length T, sequence of observation (time series)
            A : array of shape (n_state, n_state), transition matrix of the hidden states
            B : Observation matrix given by self.emission_proba_viterbi()
            -----------
            Returns : 
            x : list of length T, most likely sequence of hidden states given the data
            T1[0] : log likelihood of most likely sequence.
        """
        
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
        """ Compute the log likelihood that a new series was generated by trained self.bayesian_hmm
            -----------
            Parameters :
            new_series : list like object of floats values and arbitrary length
            -----------
            Returns : 
            float64 : log(P(new_series | model))
        """
        try: 
            self.posterior_P 
        except:
            raise Exception("Need to call fit before likelihood")
            
        B = self.emission_proba_viterbi(new_series, self.posterior_P, self.posterior_params)
        _, probas = self.viterbi(new_series, self.posterior_P, B)
        return probas[-1]
    
    
    def monitor_convergence(self): 
        """ Plots the convergence of the means and standards deviation of each states' distribution.
            Useful for diagnosis. 
            -----------
            Parameters :
            None
            -----------
            Returns : 
            None
        """
        plt.figure(figsize=(9,8))
        for s in range(self.n_states):
            plt.plot(self.cache_mu_posteriors[s], label="Group "+str(s))
        plt.legend(loc='best')
        plt.title("Convergence of means' estimators class " + str(self.class_label))
        plt.xlabel("Gibbs iterations")
        plt.show()
    
        plt.figure(figsize=(9,8))
        for s in range(self.n_states):
            plt.plot(self.cache_sigma_posteriors[s], label="Group "+str(s))
        plt.legend(loc='best')
        plt.title("Convergence of Standards Deviations' estimators class  " + str(self.class_label))
        plt.xlabel("Gibbs iterations")
        plt.show()
    
    def segmentation(self):
        """ Desc 
            -----------
            Parameters :
            None
            -----------
            Returns : 
            None
        """
        pass
