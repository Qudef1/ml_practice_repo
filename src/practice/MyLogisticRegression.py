import numpy as np


class MyLogisticRegression:
    def __init__(self, lr=0.01, n_iters=1000, lambda_l1=0.1):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = 0.001
        self.lambda_l1 = lambda_l1

    def fit(self, X, y):
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        for _ in range(self.n_iters):
            linear = X@self.weights + self.bias
            y_pred = self.sigmoid(linear)
            dw = (X.T@(y_pred-y))*(1/n_samples) + \
                np.sign(self.weights)*self.lambda_l1
            db = (np.sum(y_pred-y))*(1/n_samples)

            self.weights -= dw*self.lr
            self.bias -= db*self.lr

    def predict(self, X, hold=0.5):
        y_pred = self.sigmoid(X@self.weights+self.bias)
        return (y_pred >= hold).astype(int)

    @staticmethod
    def sigmoid(z):
        return 1 / (1+np.exp(-z))
