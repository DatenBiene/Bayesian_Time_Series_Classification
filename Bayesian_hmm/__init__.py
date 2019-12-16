""" Module for time series classification using Bayesian Hidden Markov Model 
    -----------------
    Version : 0.2
    Date : December, 11th 2019
    Authors : Mehdi Bennaceur 
    Phase : Development 
    Contact : _
    Github : https://github.com/DatenBiene/Bayesian_Time_Series_Classification
"""

__version__ = "0.2"
__date__ = "December, 11th 2019"
__author__ = 'Mehdi Bennaceur'
__github__ = "https://github.com/DatenBiene/Bayesian_Time_Series_Classification"

from Bayesian_hmm.bayes_hmm import bayesian_hmm 
from Bayesian_hmm.simulate_data import generate_markov_seq, generate_transtion_matrix, generate_series, generate_samples
from Bayesian_hmm.utils import assign_classes, build_hmm_models


__all__ = ["bayesian_hmm", "generate_markov_seq", "generate_transtion_matrix", "generate_series", "generate_samples",
          "assign_classes", "build_hmm_models"]
