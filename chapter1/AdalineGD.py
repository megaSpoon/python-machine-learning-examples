import numpy as np

class AdaLineGD(object):
    """
    Adaline Linear Gradient Descent Neuron classifier.

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

    def __init__(self, eta = 0.01, n_iter = 50):
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
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        """return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
