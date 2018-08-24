from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from os.path import join
import datetime
from hacking_180824.util.utils import random_sampling, rob_sampler
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from modAL.uncertainty import entropy_sampling

log_dir = 'log'
log_file_name = join(log_dir, datetime.datetime.now().isoformat() + '.log')

logger = logging.getLogger('hello')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_file_name)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)


def normalize(data, reverse=False):
    if reverse:
        return data * 78. + 33.
    else:
        return (data - 33.) / 78.


def load_mnist():
    """
    Load the MNIST data set
    :return:
    """
    mndata = MNIST('data/mnist')
    data = {}

    # train data
    images, labels = mndata.load_training()
    images = np.reshape(normalize(np.array(images)), newshape=[-1, 784])
    labels = np.array(labels).astype(np.int64)

    # Split the train data into a train and val set
    N = images.shape[0]
    ratio = int(0.001 * N)
    ind = np.random.permutation(N)

    data['X_train'] = images[ind[:ratio]]
    data['y_train'] = labels[ind[:ratio]]

    data['X_val'] = images[ind[ratio:]]
    data['y_val'] = labels[ind[ratio:]]

    # test data
    images, labels = mndata.load_testing()
    images = np.reshape(normalize(np.array(images)), newshape=[-1, 784])
    labels = np.array(labels).astype(np.int64)

    data['X_test'] = images
    data['y_test'] = labels

    logger.debug(f'train set {data["X_train"].shape[0]} - val set {data["X_val"].shape[0]} - test set {data["X_test"].shape[0]}')
    return data


def delete_idx(data, queries):
    data['X_val'] = np.delete(data['X_val'], queries, axis=0)
    data['y_val'] = np.delete(data['y_val'], queries, axis=0)


def main():
    data = load_mnist()

    # initializing the learner
    estimator = LogisticRegression(n_jobs=8, tol=1E-3)
    estimator = MLPClassifier(hidden_layer_sizes=(30,), activation='tanh')
    learner = ActiveLearner(
        estimator=estimator,
        X_training=data['X_train'], y_training=data['y_train'],
        query_strategy=random_sampling
    )

    # Tell here what the name of your policy is
    logger.debug('policyname --- random')

    performances = []
    num_steps = 5  # Number of steps in the active learning  loop
    t1 = time.time()
    for num_step in range(num_steps):
        # query for labels
        query_idxs, query_insts = learner.query(data['X_val'], n_instances=200)

        # Get global performance
        performance = learner.score(data['X_test'], data['y_test'])
        logger.debug(f'--- STEP {num_step:5.0f} ---'
              f'PERFORMANCE {performance:8.3f} ---'
              f'and {data["X_val"].shape} samples left in pool'
              f'in {time.time() - t1:8.5f} seconds')

        # Log global performance
        logger.info(f'PERFORMANCE ---{num_step:10.0f}---{performance:10.3f}')
        t1 = time.time()

        # Get per class performance

        # Teach the learner with new labels
        learner.teach(data['X_val'][query_idxs], data['y_val'][query_idxs])

        # Delete
        delete_idx(data, query_idxs)

        performances.append((num_step, performance))

    return np.array(performances)


if __name__ == '__main__':
    performances = main()




