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


def build_hmm_models(X_data, n_model, n_states, hmm_obj, n_iter_gibbs=2000, max_obs = None):
    print("Building", n_model, " Bayesian HMM instances with", n_states, "hidden states ...")
    #create priors 
    if max_obs is None:
        max_obs = 900
    models = []
    for classe in range(1, n_model + 1):
        print("------")
        print("Fitting Class", classe, "...")
        #sns.distplot(np.concatenate(X_data[classe])[:max_obs], hist=False)
        #plt.show()
        prior_p, prior_transitions = {}, np.ones((n_states, n_states))
        for s in range(n_states):
            prior_p[s] = [0, 1, 1, 2]
        model_class_step = hmm_obj(prior_p, prior_transitions, n_iter_gibbs, class_label=classe)
        model_class_step.fit(np.concatenate(X_data[classe])[:max_obs])
        models.append(model_class_step)
    return models
