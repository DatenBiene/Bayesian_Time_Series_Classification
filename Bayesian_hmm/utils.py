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
    for classe in X_data:
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


def train_ts_to_dic(X_train, y_train):
    X_train_dic = {i: [] for i in set(y_train)}
    for j in range(X_train.shape[0]):
        y = y_train[j]
        X_train_dic[y] += [np.array(X_train.iloc[j][0])]
    return X_train_dic


def test_ts_to_list(X_test):
    X_test_dic = []
    for j in range(X_test.shape[0]):
        X_test_dic += [np.array(X_test.iloc[j][0])]
    return X_test_dic
