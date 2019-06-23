import warnings

import numpy as np
import scipy.sparse as sp

from ..base import BaseEstimator, ClusterMixin, TransformerMixin
from ..metrics.pairwise import euclidean_distances
from ..metrics.pairwise import pairwise_distances_argmin_min
from ..utils.extmath import row_norms, squared_norm, stable_cumsum
from ..utils.sparsefuncs_fast import assign_rows_csr
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import _num_samples
from ..utils import check_array
from ..utils import gen_batches
from ..utils import check_random_state
from ..utils.validation import check_is_fitted
from ..utils.validation import FLOAT_DTYPES
from ..utils._joblib import Parallel
from ..utils._joblib import delayed
from ..utils._joblib import effective_n_jobs
from ..exceptions import ConvergenceWarning
from . import _k_means
from ._k_means_elkan import k_means_elkan

class KMeans(BaseEstimator, ClusterMixin, TransformerMixin):
    """K-Means clustering
    Read more in the :ref:`User Guide <k_means>`.
    Parameters
    ----------
    n_clusters : int, optional, default: 8
        The number of clusters to form as well as the number of
        centroids to generate.
    init : {'k-means++', 'random' or an ndarray}
        Method for initialization, defaults to 'k-means++':
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
    n_init : int, default: 10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    max_iter : int, default: 300
        Maximum number of iterations of the k-means algorithm for a
        single run.
    tol : float, default: 1e-4
        Relative tolerance with regards to inertia to declare convergence
    precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).
        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.
        True : always precompute distances
        False : never precompute distances
    verbose : int, default 0
        Verbosity mode.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    copy_x : boolean, optional
        When pre-computing distances it is more numerically accurate to center
        the data first.  If copy_x is True (default), then the original data is
        not modified, ensuring X is C-contiguous.  If False, the original data
        is modified, and put back before the function returns, but small
        numerical differences may be introduced by subtracting and then adding
        the data mean, in this case it will also not ensure that data is
        C-contiguous which may cause a significant slowdown.
    n_jobs : int or None, optional (default=None)
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.
    algorithm : "auto", "full" or "elkan", default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient by using the triangle
        inequality, but currently doesn't support sparse data. "auto" chooses
        "elkan" for dense data and "full" for sparse data.
    Attributes
    ----------
    cluster_centers_ : array, [n_clusters, n_features]
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.
    labels_ :
        Labels of each point
    inertia_ : float
        Sum of squared distances of samples to their closest cluster center.
    n_iter_ : int
        Number of iterations run.
    Examples
    --------
    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])
    See also
    --------
    MiniBatchKMeans
        Alternative online implementation that does incremental updates
        of the centers positions using mini-batches.
        For large scale learning (say n_samples > 10k) MiniBatchKMeans is
        probably much faster than the default batch implementation.
    Notes
    -----
    The k-means problem is solved using either Lloyd's or Elkan's algorithm.
    The average complexity is given by O(k n T), were n is the number of
    samples and T is the number of iteration.
    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features. (D. Arthur and S. Vassilvitskii,
    'How slow is the k-means method?' SoCG2006)
    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.
    If the algorithm stops before fully converging (because of ``tol`` or
    ``max_iter``), ``labels_`` and ``cluster_centers_`` will not be consistent,
    i.e. the ``cluster_centers_`` will not be the means of the points in each
    cluster. Also, the estimator will reassign ``labels_`` after the last
    iteration to make ``labels_`` consistent with ``predict`` on the training
    set.
    """

    def __init__(self, n_clusters=8, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='auto',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs=None, algorithm='auto'):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.algorithm = algorithm

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse='csr', dtype=FLOAT_DTYPES)
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError("Incorrect number of features. "
                             "Got %d features, expected %d" % (
                                 n_features, expected_n_features))

        return X

    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.
        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
        y : Ignored
            not used, present here for API consistency by convention.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None)
        """
        random_state = check_random_state(self.random_state)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = \
            k_means(
                X, n_clusters=self.n_clusters, sample_weight=sample_weight,
                init=self.init, n_init=self.n_init,
                max_iter=self.max_iter, verbose=self.verbose,
                precompute_distances=self.precompute_distances,
                tol=self.tol, random_state=random_state, copy_x=self.copy_x,
                n_jobs=self.n_jobs, algorithm=self.algorithm,
                return_n_iter=True)
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute cluster centers and predict cluster index for each sample.
        Convenience method; equivalent to calling fit(X) followed by
        predict(X).
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        y : Ignored
            not used, present here for API consistency by convention.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None)
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def fit_transform(self, X, y=None, sample_weight=None):
        """Compute clustering and transform X to cluster-distance space.
        Equivalent to fit(X).transform(X), but more efficiently implemented.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        y : Ignored
            not used, present here for API consistency by convention.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None)
        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        # Currently, this just skips a copy of the data if it is not in
        # np.array or CSR format already.
        # XXX This skips _check_test_data, which may change the dtype;
        # we should refactor the input validation.
        return self.fit(X, sample_weight=sample_weight)._transform(X)

    def transform(self, X):
        """Transform X to a cluster-distance space.
        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to transform.
        Returns
        -------
        X_new : array, shape [n_samples, k]
            X transformed in the new space.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        """guts of transform method; no input validation"""
        return euclidean_distances(X, self.cluster_centers_)

    def predict(self, X, sample_weight=None):
        """Predict the closest cluster each sample in X belongs to.
        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data to predict.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None)
        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        return _labels_inertia(X, sample_weight, x_squared_norms,
                               self.cluster_centers_)[0]

    def score(self, X, y=None, sample_weight=None):
        """Opposite of the value of X on the K-means objective.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            New data.
        y : Ignored
            not used, present here for API consistency by convention.
        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None)
        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self, 'cluster_centers_')

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        return -_labels_inertia(X, sample_weight, x_squared_norms,
                                self.cluster_centers_)[1]


def _mini_batch_step(X, sample_weight, x_squared_norms, centers, weight_sums,
                     old_center_buffer, compute_squared_diff,
                     distances, random_reassign=False,
                     random_state=None, reassignment_ratio=.01,
                     verbose=False):
    """Incremental update of the centers for the Minibatch K-Means algorithm.
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The original data array.
    sample_weight : array-like, shape (n_samples,)
        The weights for each observation in X.
    x_squared_norms : array, shape (n_samples,)
        Squared euclidean norm of each data point.
    centers : array, shape (k, n_features)
        The cluster centers. This array is MODIFIED IN PLACE
    counts : array, shape (k,)
         The vector in which we keep track of the numbers of elements in a
         cluster. This array is MODIFIED IN PLACE
    distances : array, dtype float, shape (n_samples), optional
        If not None, should be a pre-allocated array that will be used to store
        the distances of each sample to its closest center.
        May not be None when random_reassign is True.
    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization and to
        pick new clusters amongst observations with uniform probability. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    random_reassign : boolean, optional
        If True, centers with very low counts are randomly reassigned
        to observations.
    reassignment_ratio : float, optional
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more likely to be reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.
    verbose : bool, optional, default False
        Controls the verbosity.
    compute_squared_diff : bool
        If set to False, the squared diff computation is skipped.
    old_center_buffer : int
        Copy of old centers for monitoring convergence.
    Returns
    -------
    inertia : float
        Sum of squared distances of samples to their closest cluster center.
    squared_diff : numpy array, shape (n_clusters,)
        Squared distances between previous and updated cluster centers.
    """
    # Perform label assignment to nearest centers
    nearest_center, inertia = _labels_inertia(X, sample_weight,
                                              x_squared_norms, centers,
                                              distances=distances)

    if random_reassign and reassignment_ratio > 0:
        random_state = check_random_state(random_state)
        # Reassign clusters that have very low weight
        to_reassign = weight_sums < reassignment_ratio * weight_sums.max()
        # pick at most .5 * batch_size samples as new centers
        if to_reassign.sum() > .5 * X.shape[0]:
            indices_dont_reassign = \
                    np.argsort(weight_sums)[int(.5 * X.shape[0]):]
            to_reassign[indices_dont_reassign] = False
        n_reassigns = to_reassign.sum()
        if n_reassigns:
            # Pick new clusters amongst observations with uniform probability
            new_centers = random_state.choice(X.shape[0], replace=False,
                                              size=n_reassigns)
            if verbose:
                print("[MiniBatchKMeans] Reassigning %i cluster centers."
                      % n_reassigns)

            if sp.issparse(X) and not sp.issparse(centers):
                assign_rows_csr(
                        X, new_centers.astype(np.intp, copy=False),
                        np.where(to_reassign)[0].astype(np.intp, copy=False),
                        centers)
            else:
                centers[to_reassign] = X[new_centers]
        # reset counts of reassigned centers, but don't reset them too small
        # to avoid instant reassignment. This is a pretty dirty hack as it
        # also modifies the learning rates.
        weight_sums[to_reassign] = np.min(weight_sums[~to_reassign])

    # implementation for the sparse CSR representation completely written in
    # cython
    if sp.issparse(X):
        return inertia, _k_means._mini_batch_update_csr(
            X, sample_weight, x_squared_norms, centers, weight_sums,
            nearest_center, old_center_buffer, compute_squared_diff)

    # dense variant in mostly numpy (not as memory efficient though)
    k = centers.shape[0]
    squared_diff = 0.0
    for center_idx in range(k):
        # find points from minibatch that are assigned to this center
        center_mask = nearest_center == center_idx
        wsum = sample_weight[center_mask].sum()

        if wsum > 0:
            if compute_squared_diff:
                old_center_buffer[:] = centers[center_idx]

            # inplace remove previous count scaling
            centers[center_idx] *= weight_sums[center_idx]

            # inplace sum with new points members of this cluster
            centers[center_idx] += \
                np.sum(X[center_mask] *
                       sample_weight[center_mask, np.newaxis], axis=0)

            # update the count statistics for this center
            weight_sums[center_idx] += wsum

            # inplace rescale to compute mean of all points (old and new)
            # Note: numpy >= 1.10 does not support '/=' for the following
            # expression for a mixture of int and float (see numpy issue #6464)
            centers[center_idx] = centers[center_idx] / weight_sums[center_idx]

            # update the squared diff if necessary
            if compute_squared_diff:
                diff = centers[center_idx].ravel() - old_center_buffer.ravel()
                squared_diff += np.dot(diff, diff)

    return inertia, squared_diff


def _mini_batch_convergence(model, iteration_idx, n_iter, tol,
                            n_samples, centers_squared_diff, batch_inertia,
                            context, verbose=0):
    """Helper function to encapsulate the early stopping logic"""
    # Normalize inertia to be able to compare values when
    # batch_size changes
    batch_inertia /= model.batch_size
    centers_squared_diff /= model.batch_size

    # Compute an Exponentially Weighted Average of the squared
    # diff to monitor the convergence while discarding
    # minibatch-local stochastic variability:
    # https://en.wikipedia.org/wiki/Moving_average
    ewa_diff = context.get('ewa_diff')
    ewa_inertia = context.get('ewa_inertia')
    if ewa_diff is None:
        ewa_diff = centers_squared_diff
        ewa_inertia = batch_inertia
    else:
        alpha = float(model.batch_size) * 2.0 / (n_samples + 1)
        alpha = 1.0 if alpha > 1.0 else alpha
        ewa_diff = ewa_diff * (1 - alpha) + centers_squared_diff * alpha
        ewa_inertia = ewa_inertia * (1 - alpha) + batch_inertia * alpha

    # Log progress to be able to monitor convergence
    if verbose:
        progress_msg = (
            'Minibatch iteration %d/%d:'
            ' mean batch inertia: %f, ewa inertia: %f ' % (
                iteration_idx + 1, n_iter, batch_inertia,
                ewa_inertia))
        print(progress_msg)

    # Early stopping based on absolute tolerance on squared change of
    # centers position (using EWA smoothing)
    if tol > 0.0 and ewa_diff <= tol:
        if verbose:
            print('Converged (small centers change) at iteration %d/%d'
                  % (iteration_idx + 1, n_iter))
        return True

    # Early stopping heuristic due to lack of improvement on smoothed inertia
    ewa_inertia_min = context.get('ewa_inertia_min')
    no_improvement = context.get('no_improvement', 0)
    if ewa_inertia_min is None or ewa_inertia < ewa_inertia_min:
        no_improvement = 0
        ewa_inertia_min = ewa_inertia
    else:
        no_improvement += 1

    if (model.max_no_improvement is not None
            and no_improvement >= model.max_no_improvement):
        if verbose:
            print('Converged (lack of improvement in inertia)'
                  ' at iteration %d/%d'
                  % (iteration_idx + 1, n_iter))
        return True

    # update the convergence context to maintain state across successive calls:
    context['ewa_diff'] = ewa_diff
    context['ewa_inertia'] = ewa_inertia
    context['ewa_inertia_min'] = ewa_inertia_min
    context['no_improvement'] = no_improvement
    return False

