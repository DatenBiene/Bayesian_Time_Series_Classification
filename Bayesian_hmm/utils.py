import numpy as np

def assign_classes(series, models, verbose=False):
    """ models must be a list of bayesian_hmm fitted object"""
    predicted_labels = []
    if verbose:
        print("-----")
        print("Assigning class to series")
        print("-----")
    for new_series in series:
        probas = []
        for m in models:
            probas.append((m.likelihood(new_series), m.class_label))
        probas = sorted(probas, key=lambda x : x[0])
        if verbose:
            print(probas)
            print("----")
        predicted_labels.append(probas[-1][1])

    return np.array(predicted_labels)
