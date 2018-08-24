import numpy as np
from modAL.uncertainty import classifier_entropy


def random_sampling(classifier, X_pool, n_instances=1):
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples), size=n_instances)
    return query_idx, X_pool[query_idx]


def rob_sampler(classifier, X, n_instances=1, **uncertainty_measure_kwargs):
    entropy = classifier_entropy(classifier, X, **uncertainty_measure_kwargs)

    sample_weights = entropy / np.sum(entropy)

    query_idx = np.random.choice(range(len(entropy)), size=n_instances, p=sample_weights)
    return query_idx, X[query_idx]