from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from os.path import join
import datetime
from hacking_180824.util.utils import random_sampling, rob_sampler, jetze_sampler
from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from modAL.uncertainty import entropy_sampling, margin_sampling, uncertainty_sampling
from sklearn.metrics import confusion_matrix

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
    """
    hardcoded normalization for mnist
    :param data:
    :param reverse:
    :return:
    """
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

    # In case of biased sampling based on the labels, so that higher labels get sampled more often
    # sampling_weights = 2**labels/np.sum(2**labels)
    # biased_samples = np.random.choice(range(len(labels)), size=ratio, p=sampling_weights, replace=False)

    # In case of sampling only labels lower than 4
    biased_samples = np.random.choice(np.where(labels < 4)[0], size=ratio, replace=False)

    data['X_train'] = images[biased_samples]
    data['y_train'] = labels[biased_samples]

    data['X_pool'] = images[np.delete(np.arange(N), biased_samples)]
    data['y_pool'] = labels[np.delete(np.arange(N), biased_samples)]

    # test data
    images, labels = mndata.load_testing()
    images = np.reshape(normalize(np.array(images)), newshape=[-1, 784])
    labels = np.array(labels).astype(np.int64)

    data['X_test'] = images
    data['y_test'] = labels

    logger.debug(f'train set {data["X_train"].shape[0]} - '
                 f'val set {data["X_pool"].shape[0]} - '
                 f'test set {data["X_test"].shape[0]}')
    return data


def delete_idx(data, queries):
    data['X_pool'] = np.delete(data['X_pool'], queries, axis=0)
    data['y_pool'] = np.delete(data['y_pool'], queries, axis=0)


def main():
    data = load_mnist()

    # Keep track of the number of instances per class
    counts_cum = np.array([np.sum(data['y_train'] == num) for num in range(10)])

    # Initializing the learner
    # simple use sklearn estimators
    estimator = LogisticRegression(n_jobs=8, tol=1E-3)
    # estimator = MLPClassifier(hidden_layer_sizes=(30,), activation='tanh')

    # Set up the learner. This object will maintain the data
    # and decide which points to query according to query_strategy
    learner = ActiveLearner(
        estimator=estimator,
        X_training=data['X_train'], y_training=data['y_train'],
        query_strategy=rob_sampler
    )

    # Tell here what the name of your policy is
    # We will use this for plotting in `plotting/main_plot.py`
    logger.debug('policyname --- margin_sampling')

    num_steps = 20  # Number of steps in the active learning  loop
    t1 = time.time()
    for num_step in range(num_steps):
        # query for labels
        # Finds the n_instances most informative point in the data provided by calling
        #         the query_strategy function. Returns the queried instances and its indices.
        query_idxs, query_insts = learner.query(data['X_pool'], n_instances=20)

        # Get global performance
        # This follows the SKlearn interface for scoring
        performance = learner.score(data['X_test'], data['y_test'])
        logger.debug(f'--- STEP {num_step:5.0f} ---'
              f'PERFORMANCE {performance:8.3f} ---'
              f'and {data["X_pool"].shape} samples left in pool'
              f'in {time.time() - t1:8.5f} seconds')

        # Log global performance
        logger.info(f'PERFORMANCE ---{num_step:10.0f}---{performance:10.3f}')
        t1 = time.time()

        # Get per class performance
        y_test_pred = learner.predict(data['X_test'])
        C = confusion_matrix(data['y_test'], y_test_pred)
        #    sum over the axis=0 in the following line to know the accuracy per true class
        per_class_accuracy = np.diag(C) / (np.sum(C, axis=0) + 1E-9)  # Add small number to avoid numerical error

        # Log the per class performance
        logger.info(f'PERCLASS ---{num_step:10.0f}--- ' + '--'.join((f'{float(p):.3f}' for p in per_class_accuracy)))

        # Get the per class counts
        counts = [np.sum(data['y_pool'][query_idxs] == num) for num in range(10)]
        counts_cum += np.array(counts)

        # Log the per class counts
        logger.info(f'COUNTS ---{num_step:10.0f}---' + '--'.join([str(int(c)) for c in counts_cum]))

        # Teach the learner with new labels
        learner.teach(data['X_pool'][query_idxs], data['y_pool'][query_idxs])

        # Delete the queried data as they are ingested already by the learner
        delete_idx(data, query_idxs)


if __name__ == '__main__':
   main()




