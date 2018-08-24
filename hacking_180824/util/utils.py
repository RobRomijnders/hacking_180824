import numpy as np
from modAL.uncertainty import classifier_entropy, classifier_margin


def random_sampling(classifier, X_pool, n_instances=1):
    """
    Simply sample random indices, without using classifier performance
    :param classifier:
    :param X_pool:
    :param n_instances:
    :return:
    """
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples), size=n_instances)
    return query_idx, X_pool[query_idx]


def rob_sampler(classifier, X, n_instances=1, **uncertainty_measure_kwargs):
    """
    Sample randomly with sample weights based on entropy
    :param classifier:
    :param X:
    :param n_instances:
    :param uncertainty_measure_kwargs:
    :return:
    """
    entropy = classifier_entropy(classifier, X, **uncertainty_measure_kwargs)

    sample_weights = entropy / np.sum(entropy)

    query_idx = np.random.choice(range(len(entropy)), size=n_instances, p=sample_weights)
    return query_idx, X[query_idx]


def jetze_sampler(classifier, X, n_instances=1, **uncertainty_measure_kwargs):
    """
    Sample randomly with weights based on the margin
    :param classifier:
    :param X:
    :param n_instances:
    :param uncertainty_measure_kwargs:
    :return:
    """
    margin = classifier_margin(classifier, X, **uncertainty_measure_kwargs)

    sample_weights = margin / np.sum(margin)

    query_idx = np.random.choice(range(len(margin)), size=n_instances, p=sample_weights)
    return query_idx, X[query_idx]