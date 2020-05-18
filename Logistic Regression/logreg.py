import numpy as np
import seaborn as sns


def compute_cost(X, y, params):
    n_samples = len(y)
    h = sigmoid( X @ params)
    eps = 1e-5
    cost = (1 / n_samples) * (  ((-y).T @ np.log(h + eps))  - ((1-y).T @ np.log(1 - h + eps)) )
    return cost

def gradient_descent(X, y, params, learning_rate, _iter):
    n_samples = len(y)
    history = np.zeros((_iter, 1))
    
    for i in range(_iter):
        params = params - (learning_rate / n_samples) * (X.T @ (sigmoid(X @ params) - y))
        history[i] = compute_cost(X, y, params)
        
    return cost_history, parms


class LogisticRegression():
    def __init__(self, X, y, alpha, n_iter):
        self.alpha = alpha
        self.n_iter = n_iter
        self.y = y.reshape(-1, 1)
        self.n_samples = len(y)
        self.X = np.hstack((np.ones(self.n_samples, 1)), X) # fit intercept
        self.params = np.zeros(self.n_samples + 1 , 1)
        self._coef = None
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict_proba(self, X):
        return sigmoid( X @ self.params)

    def predict(self, X, thresh = .5):
        return predict_proba(X) >= thresh
    
    def fit(self):
        for i in range(self.n_iter):
            self.params = self.params - (self.alpha / self.n_samples) * \
                            (self.X.T @ (sigmoid(self.X @ self.params) - y) )
         
        
        return self

    
        