""" Module for time series classification using Bayesian Hidden Markov Model 
    -----------------
    Version : 0.1 
    Date : December, 11th 2019
    Authors : Pierre Delanoue & Mehdi Bennaceur 
    Phase : Development 
    Contact : _
    Github : https://github.com/DatenBiene/Bayesian_Time_Series_Classification
"""


from Bayesian_hmm.bayes_hmm import bayesian_hmm 
from Bayesian_hmm.simulate_data import generate_markov_seq, generate_transtion_matrix, generate_series, generate_samples
from Bayesian_hmm.utils import assign_classes


__all__ = ["bayesian_hmm", "generate_markov_seq", "generate_transtion_matrix", "generate_series", "generate_samples",
          "assign_classes"]
