import numpy as np

class LinearRegression():
    def __init__(self, X, y, alpha = .03, n_iter = 1200):
        self.alpha = alpha
        self.n_samples = len(y)
        self.X = np.hstack( np.ones(self.n_samples, 1), (X - np.mean(X, 0)) / np.std(X, 0))
        self.y = y.reshape(-1, 1)
        self.n_iter = n_iter
        self.params = np.zeros((self.n_samples + 1, 1) ) # +1 for intercept
        self._coef = None
        self._intercept = None
    
    def fit(self):
        for i in range(self.n_iter):
            self.params = self.params - (self.alpha/self.n_samples) * \
                                self.X.T @ (self.X @ self.params - self.y)
            
        self._intercept = self.params[0]
        self._coef = self.params[1:]
        
        return self
    
    def score(self, X = None, y = None):
        if X is None:
            X = self.X
        else:
            n_samples = np.size(X, 0)
            X = np.hstack( (np.ones(n_samples, 1)), (X - np.mean(X, 0))/np.std(X, 0) )
        
        if y is None:
            y = self.y
        else:
            y = y.reshape(-1, 1)
        
        y_pred = X @ self.params
        score = 1 - (((y - y_pred)**2).sum() / ((y - y.mean())**2).sum())
        
        return score

    def predict(self, X):
        n_samples = X.shape[0]
        y = np.hstack( (np.ones(n_samples, 1 )), 
                     (X - np.mean(X, 0)) / np.std(X, 0)) @ self.params
        return y

    def get_params(self):
        return self.params
        