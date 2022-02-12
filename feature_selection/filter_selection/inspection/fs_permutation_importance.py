"""Permutation importance for estimators."""
import lightgbm
import numpy as np
import ray
from sklearn.metrics import check_scoring
from sklearn.utils import Bunch
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from sklearn.utils.validation import _deprecate_positional_args


def _weights_scorer(scorer, estimator, X, y, sample_weight):
    # ray_para = RayParams(num_actors=2, cpus_per_actor=2)
    if sample_weight is not None:
        return scorer(estimator, X, y, sample_weight)
    return scorer(estimator, X, y)


def _calculate_permutation_scores_per_col(estimator, X, y, sample_weight, col_idx,
                                          random_state, n_repeats, scorer):
    """Calculate score when `col_idx` is permuted."""
    random_state = check_random_state(random_state)

    # Work on a copy of X to to ensure thread-safety in case of threading based
    # parallelism. Furthermore, making a copy is also useful when the joblib
    # backend is 'loky' (default) or the old 'multiprocessing': in those cases,
    # if X is large it will be automatically be backed by a readonly memory map
    # (memmap). X.copy() on the other hand is always guaranteed to return a
    # writable data-structure whose columns can be shuffled inplace.
    X_permuted = X.copy()
    scores = np.zeros(n_repeats)
    shuffling_idx = np.arange(X.shape[0])
    for n_round in range(n_repeats):
        random_state.shuffle(shuffling_idx)
        if hasattr(X_permuted, "iloc"):
            col = X_permuted.iloc[shuffling_idx, col_idx]
            col.index = X_permuted.index
            X_permuted.iloc[:, col_idx] = col
        else:
            X_permuted[:, col_idx] = X_permuted[shuffling_idx, col_idx]
        feature_score = _weights_scorer(
            scorer, estimator, X_permuted, y, sample_weight
        )
        scores[n_round] = feature_score

    return scores


@ray.remote
def _calculate_permutation_scores(estimator, X, y, sample_weight, col_idx,
                                  random_state, n_repeats, scorer):
    """Calculate score when `col_idx` is permuted."""
    print("计算：{}-{}列".format(col_idx[0], col_idx[-1]))
    print("type.X: {}".format(type(X)))
    scores = []
    if len(col_idx) == 0:
        return scores
    for i in range(len(col_idx)):
        scores.append(_calculate_permutation_scores_per_col(estimator, X, y, sample_weight, col_idx[i], random_state,
                                                            n_repeats, scorer))
    return scores


def predict(booster, x):
    result = booster.predict(x)
    if len(result.shape) < 2:
        result = np.vstack((1. - result, result)).transpose()
    # print("result shape:{}".format(result.shape))
    class_index = np.argmax(result, axis=1)
    # print("class_index shape:{}".format(class_index.shape))
    return class_index


def classify_booster_score(booster: lightgbm.basic.Booster, x, y, sample_weight=None):
    """
    Return the mean accuracy on the given test data and labels.

    In multi-label classification, this is the subset accuracy
    which is a harsh metric since you require for each sample that
    each label set be correctly predicted.

    Parameters
    ----------
    booster : lightgbm.booster

    x : array-like of shape (n_samples, n_features)
        Test samples.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True labels for `X`.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    score : float
        Mean accuracy of ``self.predict(X)`` wrt. `y`.
    """
    from sklearn.metrics import accuracy_score
    return accuracy_score(y, predict(booster, x), sample_weight=sample_weight)


def regression_booster_score(booster, X, y, sample_weight=None):
    """Return the coefficient of determination :math:`R^2` of the
    prediction.

    The coefficient :math:`R^2` is defined as :math:`(1 - \\frac{u}{v})`,
    where :math:`u` is the residual sum of squares ``((y_true - y_pred)
    ** 2).sum()`` and :math:`v` is the total sum of squares ``((y_true -
    y_true.mean()) ** 2).sum()``. The best possible score is 1.0 and it
    can be negative (because the model can be arbitrarily worse). A
    constant model that always predicts the expected value of `y`,
    disregarding the input features, would get a :math:`R^2` score of
    0.0.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Test samples. For some estimators this may be a precomputed
        kernel matrix or a list of generic objects instead with shape
        ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted``
        is the number of samples used in the fitting for the estimator.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        True values for `X`.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights.

    Returns
    -------
    score : float
        :math:`R^2` of ``self.predict(X)`` wrt. `y`.

    Notes
    -----
    The :math:`R^2` score used when calling ``score`` on a regressor uses
    ``multioutput='uniform_average'`` from version 0.23 to keep consistent
    with default value of :func:`~sklearn.metrics.r2_score`.
    This influences the ``score`` method of all the multioutput
    regressors (except for
    :class:`~sklearn.multioutput.MultiOutputRegressor`).
    """

    from sklearn.metrics import r2_score
    y_pred = booster.predict(X)
    return r2_score(y, y_pred, sample_weight=sample_weight)


@_deprecate_positional_args
def permutation_importance(estimator, X, y, *, scoring=None, n_repeats=5,
                           n_jobs=2, random_state=None, sample_weight=None,
                           objective="binary"):
    """Permutation importance for feature evaluation [BRE]_.

    The :term:`estimator` is required to be a fitted estimator. `X` can be the
    data set used to train the estimator or a hold-out set. The permutation
    importance of a feature is calculated as follows. First, a baseline metric,
    defined by :term:`scoring`, is evaluated on a (potentially different)
    dataset defined by the `X`. Next, a feature column from the validation set
    is permuted and the metric is evaluated again. The permutation importance
    is defined to be the difference between the baseline metric and metric from
    permutating the feature column.

    Read more in the :ref:`User Guide <permutation_importance>`.

    Parameters
    ----------
    estimator : object
        An estimator that has already been :term:`fitted` and is compatible
        with :term:`scorer`.

    X : ndarray or DataFrame, shape (n_samples, n_features)
        Data on which permutation importance will be computed.

    y : array-like or None, shape (n_samples, ) or (n_samples, n_classes)
        Targets for supervised or `None` for unsupervised.

    scoring : string, callable or None, default=None
        Scorer to use. It can be a single
        string (see :ref:`scoring_parameter`) or a callable (see
        :ref:`scoring`). If None, the estimator's default scorer is used.

    n_repeats : int, default=5
        Number of times to permute a feature.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel. The computation is done by computing
        permutation score for each columns and parallelized over the columns.
        `None` means 1 unless in a :obj:`joblib.parallel_backend` context.
        `-1` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance, default=None
        Pseudo-random number generator to control the permutations of each
        feature.
        Pass an int to get reproducible results across function calls.
        See :term: `Glossary <random_state>`.

    sample_weight : array-like of shape (n_samples,), default=None
        Sample weights used in scoring.

        .. versionadded:: 0.24

    Returns
    -------
    result : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        importances_mean : ndarray, shape (n_features, )
            Mean of feature importance over `n_repeats`.
        importances_std : ndarray, shape (n_features, )
            Standard deviation over `n_repeats`.
        importances : ndarray, shape (n_features, n_repeats)
            Raw permutation importance scores.

    References
    ----------
    .. [BRE] L. Breiman, "Random Forests", Machine Learning, 45(1), 5-32,
             2001. https://doi.org/10.1023/A:1010933404324

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.inspection import permutation_importance
    >>> X = [[1, 9, 9],[1, 9, 9],[1, 9, 9],
    ...      [0, 9, 9],[0, 9, 9],[0, 9, 9]]
    >>> y = [1, 1, 1, 0, 0, 0]
    >>> clf = LogisticRegression().fit(X, y)
    >>> result = permutation_importance(clf, X, y, n_repeats=10,
    ...                                 random_state=0)
    >>> result.importances_mean
    array([0.4666..., 0.       , 0.       ])
    >>> result.importances_std
    array([0.2211..., 0.       , 0.       ])
    """
    if not hasattr(X, "iloc"):
        X = check_array(X, force_all_finite='allow-nan', dtype=None)

    # Precompute random seed from the random state to be used
    # to get a fresh independent RandomState instance for each
    # parallel call to _calculate_permutation_scores, irrespective of
    # the fact that variables are shared or not depending on the active
    # joblib backend (sequential, thread-based or process-based).
    random_state = check_random_state(random_state)
    random_seed = random_state.randint(np.iinfo(np.int32).max + 1)

    # scorer = check_scoring(estimator, scoring=scoring)
    if isinstance(estimator, lightgbm.basic.Booster):
        if objective == "binary" or objective == "multi_class":
            scorer = classify_booster_score
        elif objective == "regression":
            scorer = regression_booster_score
        else:
            raise NotImplementedError("目前 'objective' 仅支持 'binary', 'multi_class', 'regression' 中的一种")
    else:
        scorer = check_scoring(estimator, scoring=scoring)

    baseline_score = _weights_scorer(scorer, estimator, X, y, sample_weight)

    """scores = Parallel(n_jobs=n_jobs)(delayed(_calculate_permutation_scores)(
        estimator, X, y, sample_weight, col_idx, random_seed, n_repeats, scorer
    ) for col_idx in range(X.shape[1]))"""
    scores = []
    columns = X.shape[1]
    col_idx = [i for i in range(columns)]

    print("type.X: {}".format(type(X)))
    if columns <= n_jobs:
        permutation_scores = _calculate_permutation_scores.remote(
            estimator, X, y, sample_weight, col_idx, random_seed, n_repeats, scorer
        )
        scores = ray.get(permutation_scores)
    else:
        n_part = int(columns / n_jobs)
        permutation_scores = []
        for i in range(n_jobs):
            if i == n_jobs - 1:
                permutation_scores.append(_calculate_permutation_scores.remote(
                    estimator, X, y, sample_weight, col_idx[i * n_part:], random_seed,
                    n_repeats, scorer))
            else:
                permutation_scores.append(_calculate_permutation_scores.remote(
                    estimator, X, y, sample_weight, col_idx[i * n_part: (i + 1) * n_part], random_seed,
                    n_repeats, scorer))

        for per_score in ray.get(permutation_scores):
            scores.extend(per_score)

    print("ray.get:{}".format(scores))

    importances = baseline_score - np.array(scores)
    return Bunch(importances_mean=np.mean(importances, axis=1),
                 importances_std=np.std(importances, axis=1),
                 importances=importances)
