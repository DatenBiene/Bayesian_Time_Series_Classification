"""
    Bayesian KNN time series classification built on sktime and sklearn KNeighborsClassifier
    Adapted from KNeighborsTimeSeriesClassifier from sktime (author: Jason Lines)
"""

__author__ = "Pierre Delanoue"

from scipy import stats
from sklearn.utils.extmath import weighted_mode
from sklearn.neighbors.classification import KNeighborsClassifier
from functools import partial
import warnings
import numpy as np
from sklearn.metrics import pairwise_distances_chunked
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import effective_n_jobs
from sklearn.exceptions import DataConversionWarning
from sktime.distances.elastic_cython import dtw_distance, wdtw_distance, ddtw_distance, wddtw_distance, lcss_distance, erp_distance, msm_distance, twe_distance
from sktime.distances.mpdist import mpdist
from sklearn.utils.validation import check_array
from sklearn.neighbors.base import _check_weights, _get_weights
import pandas as pd
from numba import jit


class BayesianNeighborsTimeSeriesClassifier(KNeighborsClassifier):
    """
    An adapted version of the sktime KNeighborsTimeSeriesClassifier.
    The Bayesian approach automatically selects a number of neighbours k for each observation.

    Parameters
    ----------
    n_neighbors_bayes   : int, number of neighbors to consider
    p_gamma             : a priori parameter of the geometric law for k
    weights             : mechanism for weighting a vote: 'uniform', 'distance' or a callable function: default ==' uniform'
    algorithm           : search method for neighbours {‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}: default = 'brute'
    metric              : distance measure for time series: {'dtw','ddtw','wdtw','lcss','erp','msm','twe'}: default ='dtw'
    metric_params       : dictionary for metric parameters: default = None
    """
    def __init__(self, n_neighbors_bayes=100, p_gamma=1/20, weights='uniform', algorithm='brute', metric='dtw', metric_params=None, **kwargs):

        self._cv_for_params = False

        if metric == 'dtw':
            metric = dtw_distance
        elif metric == 'ddtw':
            metric = ddtw_distance
        elif metric == 'wdtw':
            metric = wdtw_distance
        elif metric == 'wddtw':
            metric = wddtw_distance
        elif metric == 'lcss':
            metric = lcss_distance
        elif metric == 'erp':
            metric = erp_distance
        elif metric == 'msm':
            metric = msm_distance
        elif metric == 'twe':
            metric = twe_distance
        elif metric == 'mpdist':
            metric = mpdist

        else:
            if type(metric) is str:
                raise ValueError("Unrecognised distance measure: "+metric+". Allowed values are names from [dtw,ddtw,wdtw,wddtw,lcss,erp,msm] or "
                                                                          "please pass a callable distance measure into the constuctor directly.")

        super().__init__(
            n_neighbors=n_neighbors_bayes,
            algorithm=algorithm,
            metric=metric,
            metric_params=metric_params,
            **kwargs)
        self.weights = _check_weights(weights)
        self.p_gamma = p_gamma

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values
        Parameters
        ----------
        X : sktime-format pandas dataframe with shape([n_cases,n_dimensions]),
        or numpy ndarray with shape([n_cases,n_readings,n_dimensions])
        y : {array-like, sparse matrix}
            Target values of shape = [n_samples]
        """
        X = check_data_sktime_tsc(X)
        self.y_train = y

        if y.ndim == 1 or y.ndim == 2 and y.shape[1] == 1:
            if y.ndim != 1:
                warnings.warn("A column-vector y was passed when a 1d array "
                              "was expected. Please change the shape of y to "
                              "(n_samples, ), for example using ravel().",
                              DataConversionWarning, stacklevel=2)

            self.outputs_2d_ = False
            y = y.reshape((-1, 1))
        else:
            self.outputs_2d_ = True

        check_classification_targets(y)
        self.classes_ = []
        self._y = np.empty(y.shape, dtype=np.int)
        for k in range(self._y.shape[1]):
            classes, self._y[:, k] = np.unique(y[:, k], return_inverse=True)
            self.classes_.append(classes)

        if not self.outputs_2d_:
            self.classes_ = self.classes_[0]
            self._y = self._y.ravel()

        temp = check_array.__code__
        check_array.__code__ = _check_array_ts.__code__
        fx = self._fit(X)
        check_array.__code__ = temp
        self.n_neighbors = min(len(y), self.n_neighbors)
        return fx

    def kneighbors(self, X, n_neighbors=None, return_distance=True):

        """Finds the K-neighbors of a point.
        Returns indices of and distances to the neighbors of each point.
        Parameters
        ----------
        X : sktime-format pandas dataframe with shape([n_cases,n_dimensions]),
        or numpy ndarray with shape([n_cases,n_readings,n_dimensions])
        y : {array-like, sparse matrix}
            Target values of shape = [n_samples]
        n_neighbors : int
            Number of neighbors to get (default is the value
            passed to the constructor).
        return_distance : boolean, optional. Defaults to True.
            If False, distances will not be returned
        Returns
        -------
        dist : array
            Array representing the lengths to points, only present if
            return_distance=True
        ind : array
            Indices of the nearest points in the population matrix.
        Examples
        --------
        In the following example, we construct a NeighborsClassifier
        class from an array representing our data set and ask who's
        the closest point to [1,1,1]
        >>> samples = [[0., 0., 0.], [0., .5, 0.], [1., 1., .5]]
        >>> from sklearn.neighbors import NearestNeighbors
        >>> neigh = NearestNeighbors(n_neighbors=1)
        >>> neigh.fit(samples) # doctest: +ELLIPSIS
        NearestNeighbors(algorithm='auto', leaf_size=30, ...)
        >>> print(neigh.kneighbors([[1., 1., 1.]])) # doctest: +ELLIPSIS
        (array([[0.5]]), array([[2]]))
        As you can see, it returns [[0.5]], and [[2]], which means that the
        element is at distance 0.5 and is the third element of samples
        (indexes start at 0). You can also query for multiple points:
        >>> X = [[0., 1., 0.], [1., 0., 1.]]
        >>> neigh.kneighbors(X, return_distance=False) # doctest: +ELLIPSIS
        array([[1],
               [2]]...)
        """
        # changed here
        check_data_sktime_tsc(X)
        check_is_fitted(self, "_fit_method")

        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        elif n_neighbors <= 0:
            raise ValueError(
                "Expected n_neighbors > 0. Got %d" %
                n_neighbors
            )
        else:
            if not np.issubdtype(type(n_neighbors), np.integer):
                raise TypeError(
                    "n_neighbors does not take %s value, "
                    "enter integer value" %
                    type(n_neighbors))

        if X is not None:
            query_is_train = False
            X = check_array(X, accept_sparse='csr', allow_nd=True)
        else:
            query_is_train = True
            X = self._fit_X
            # Include an extra neighbor to account for the sample itself being
            # returned, which is removed later
            n_neighbors += 1

        train_size = self._fit_X.shape[0]
        if n_neighbors > train_size:
            raise ValueError(
                "Expected n_neighbors <= n_samples, "
                " but n_samples = %d, n_neighbors = %d" %
                (train_size, n_neighbors)
            )

        n_samples = X.shape[0]
        sample_range = np.arange(n_samples)[:, None]

        n_jobs = effective_n_jobs(self.n_jobs)
        if self._fit_method == 'brute':

            reduce_func = partial(self._kneighbors_reduce_func,
                                  n_neighbors=n_neighbors,
                                  return_distance=return_distance)

            # for efficiency, use squared euclidean distances
            kwds = ({'squared': True} if self.effective_metric_ == 'euclidean'
                    else self.effective_metric_params_)

            result = pairwise_distances_chunked(
                X, self._fit_X, reduce_func=reduce_func,
                metric=self.effective_metric_, n_jobs=n_jobs,
                **kwds)

        else:
            raise ValueError("internal: _fit_method not recognized")

        if return_distance:
            dist, neigh_ind = zip(*result)
            result = np.vstack(dist), np.vstack(neigh_ind)
        else:
            result = np.vstack(result)

        if not query_is_train:
            return result
        else:
            # If the query data is the same as the indexed data, we would like
            # to ignore the first nearest neighbor of every sample, i.e
            # the sample itself.
            if return_distance:
                dist, neigh_ind = result
            else:
                neigh_ind = result

            sample_mask = neigh_ind != sample_range

            # Corner case: When the number of duplicates are more
            # than the number of neighbors, the first NN will not
            # be the sample, but a duplicate.
            # In that case mask the first duplicate.
            dup_gr_nbrs = np.all(sample_mask, axis=1)
            sample_mask[:, 0][dup_gr_nbrs] = False

            neigh_ind = np.reshape(
                neigh_ind[sample_mask], (n_samples, n_neighbors - 1))

            if return_distance:
                dist = np.reshape(
                    dist[sample_mask], (n_samples, n_neighbors - 1))
                return dist, neigh_ind
            return neigh_ind

    def predict_singel_k(self, X, n_neighbors=None):

        """Predict the class labels for the provided data for the selected number of neighbors.
        Parameters
        ----------
        X : sktime-format pandas dataframe or array-like, shape (n_query, n_features), \
                or (n_query, n_indexed) if metric == 'precomputed'
            Test samples.
        n_neighbors : int, number of neighbor
        Returns
        -------
        y : array of shape [n_samples] or [n_samples, n_outputs]
            Class labels for each data sample.
        """
        X = check_data_sktime_tsc(X)
        temp = check_array.__code__
        check_array.__code__ = _check_array_ts.__code__

        if n_neighbors is None:
            neigh_dist, neigh_ind = self.kneighbors(X)

        else:
            neigh_dist, neigh_ind = self.kneighbors(X, n_neighbors=n_neighbors)

        classes_ = self.classes_
        _y = self._y
        if not self.outputs_2d_:
            _y = self._y.reshape((-1, 1))
            classes_ = [self.classes_]

        n_outputs = len(classes_)
        n_samples = X.shape[0]
        weights = _get_weights(neigh_dist, self.weights)

        y_pred = np.empty((n_samples, n_outputs), dtype=classes_[0].dtype)
        for k, classes_k in enumerate(classes_):
            if weights is None:
                mode, _ = stats.mode(_y[neigh_ind, k], axis=1)
            else:
                mode, _ = weighted_mode(_y[neigh_ind, k], weights, axis=1)

            mode = np.asarray(mode.ravel(), dtype=np.intp)
            y_pred[:, k] = classes_k.take(mode)

        if not self.outputs_2d_:
            y_pred = y_pred.ravel()

        check_array.__code__ = temp
        return y_pred

    def predict(self, X):
        X = check_data_sktime_tsc(X)
        check_array.__code__ = _check_array_ts.__code__
        X = check_array(X)

        n_neighbors_bayes = self.n_neighbors

        ordered = self.kneighbors(
                                    X,
                                    n_neighbors=n_neighbors_bayes,
                                    return_distance=False
                                    )

        y_train = self.y_train

        classes, y_pos = np.unique(
                                    np.flip(y_train[ordered[0]]),
                                    return_inverse=True
                                    )
        q_posteriori = get_bayesian_k_posteriori(classes, y_pos,
                                                 p_gamma=self.p_gamma)
        k_bayes = int(np.sum(np.array(range(len(q_posteriori)))*q_posteriori))
        y_pred = self.predict_singel_k(X[[0]], k_bayes)

        l_k = [k_bayes]
        for i in range(X.shape[0]-1):
            idx = i+1
            classes, y_pos = np.unique(np.flip(y_train[ordered[idx]]),
                                       return_inverse=True)
            q_posteriori = get_bayesian_k_posteriori(classes, y_pos)
            k_bayes = int(np.sum(np.array(range(len(q_posteriori)))*q_posteriori))
            # k_bayes = np.max([np.argmax(q_posteriori), 1])

            pred = self.predict_singel_k(X[[idx]], k_bayes)

            y_pred = np.concatenate([y_pred, pred])
            l_k += [k_bayes]
        return y_pred


def check_data_sktime_tsc(X):
    """ A utility method to check the input of a TSC KNN classifier. The data must either be in
            a)  the standard sktime format (pandas dataframe with n rows and d columns for n cases with d dimesions)
            OR
            b)  a numpy ndarray with shape([n_cases,n_readings,n_dimensions]) to match the expected format for cython
                distance mesures.
        If the data matches a it will be transformed into b) and returned. If it is already in b), the input X will be
        returned without modification.
        Parameters
        -------
        X : sktime-format pandas dataframe with shape([n_cases,n_dimensions]),
        or numpy ndarray with shape([n_cases,n_readings,n_dimensions])
        y : {array-like, sparse matrix}
            Target values of shape = [n_samples]
        dim_to_use: indesx of the dimension to use (defaults to 0, i.e. first dimension)
        Returns
        -------
        X : numpy ndarray with shape([n_cases,n_readings,n_dimensions])
    """
    if type(X) is pd.DataFrame:
        if X.shape[1] > 1:
            raise TypeError("This classifier currently only supports univariate time series")
        X = np.array([np.asarray([x]).reshape(len(x), 1) for x in X.iloc[:, 0]])
    elif type(X) == np.ndarray:
        try:
            num_cases, num_readings, n_dimensions = X.shape
        except ValueError:
            raise ValueError("X should be a numpy array with 3 dimensions "
                             "([n_cases,n_readings,n_dimensions]). Instead, found: " + str(X.shape))
    return X


def _check_array_ts(array, accept_sparse=False, accept_large_sparse=True,
                    dtype="numeric", order=None, copy=False,
                    force_all_finite=True, ensure_2d=True, allow_nd=True,
                    ensure_min_samples=1, ensure_min_features=1,
                    warn_on_dtype=False, estimator=None):
    return array


@jit(nopython=True)
def get_bayesian_k_posteriori(classes, y_pos, eta_prior=None, p_gamma=1/5):
    tau = len(y_pos)
    pi = np.zeros(tau)

    growth_prop_1 = np.ones(tau + 1)
    growth_prop_2 = np.zeros(tau + 1)
    nb_classes = len(classes)

    if eta_prior is None:
        eta = np.array([10]*nb_classes)
    else:
        eta = eta_prior

    s_alpha_prior = np.sum(eta)

    for t in range(tau):
        y_t = y_pos[t]
        # will also be the indice of the corresponding alpha in eta

        for i in range(t+1):

            pi[i] = (eta[y_t] + np.sum((y_pos[:i] == y_t)*1))/(s_alpha_prior +i)
            growth_prop_2[i + 1] = growth_prop_1[i]*pi[i]*(1-p_gamma)

        pi[0] = eta[y_t]/s_alpha_prior
        growth_prop_2[0] = np.sum(growth_prop_1)*pi[0]*p_gamma

        p_evidence = np.sum(growth_prop_2)

        growth_prop_1 = growth_prop_2/p_evidence
        growth_prop_2 = np.zeros(tau + 1)

    return growth_prop_1
