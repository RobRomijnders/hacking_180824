from mnist import MNIST
import numpy as np

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
    mndata = MNIST('/home/rob/Dropbox/ml_projects/weight_uncertainty/weight_uncertainty/data/mnist')
    data = {}

    # train data
    images, labels = mndata.load_training()
    images = np.reshape(normalize(np.array(images)), newshape=[-1, 784])
    labels = np.array(labels).astype(np.int64)

    # Split the train data into a train and val set
    N = images.shape[0]
    ratio = int(0.01 * N)
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

    print(f'train set {data["X_train"].shape[0]} - val set {data["X_val"].shape[0]} - test set {data["X_test"].shape[0]}')
    return data


if __name__ == '__main__':
    data = load_mnist()

    from modAL.models import ActiveLearner
    from sklearn.ensemble import RandomForestClassifier
    from modAL.uncertainty import entropy_sampling

    # initializing the learner
    learner = ActiveLearner(
        estimator=RandomForestClassifier(),
        X_training=data['X_train'], y_training=data['y_train'],
        query_strategy=entropy_sampling
    )

    for _ in range(10000):
        # query for labels
        query_idx, query_inst = learner.query(data['X_val'])

        # ...obtaining new labels from the Oracle...

        # supply label for queried instance
        learner.teach(data['X_val'][query_idx], data['y_val'][query_idx])

        # print performance after query
        performance = learner.score(data['X_test'], data['y_test'])
        print(f'Query index is {int(query_idx):8.0f} and performance is {performance:8.3f}')

