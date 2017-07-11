import numpy as np

class Perceptron(object):
    """
    Model hyperparameters
    -----------------------
    eta: float
        learning rate (between 0.0 and 1.0)
    n_iter: int
        nums of epochs ove training dataset

    Attributes
    -----------------------
    w_: 1d-array
        weights after fitting
    errors_: list
        number of misclassfication in every epoch
    """

    def __init__(self, eta = 0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fit Training Data
        :param X: {array-like}, shape = {n_samples, n_features}
                training vectors, where n_samples is the number of samples, n_features is the number of features.
        :param y: array-like, shape = {n_samples}
                target values
        :return: self : object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for Xi, target in zip(X, y):
                update = self.eta * (target - self.predict(Xi))
                self.w_[1:] += update * Xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)

