from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from scipy import linalg

class CommonSpatialPattern(BaseEstimator, TransformerMixin):
    def __init__(self, n_components):

        if n_components % 2:
            raise ValueError(f"n_components should be even number! Got " + n_components)

        self.n_components  = n_components
    
    def _compute_average_covariance_matrix(self, X_class):
        """Compute average normalized covariance matrix for one class"""

        n_trials, n_channels, n_times = X_class.shape

        cov_sum = np.zeros((n_channels, n_channels))

        for i in range(n_trials):
            trial = X_class[i]

            #computes covariance matrix
            cov = np.cov(trial)

            #normalize by trace (sum of diagonal elements)
            cov = cov / np.trace(cov)
            cov_sum += cov
        
        cov_avg = cov_sum / n_trials
        return cov_avg

            
    def fit(self, X, y):
        
        #check only 2 classes present
        classes = np.unique(y)
        assert len(classes) == 2

        X_class0 = X[y == classes[0]]
        X_class1 = X[y == classes[1]]

        #compute Covariance matrices for both
        cov0 = self._compute_average_covariance_matrix(X_class=X_class0)
        cov1 = self._compute_average_covariance_matrix(X_class=X_class1)

        # Solve generalized eigenvalue problem
        # Find W such that: cov_class0 @ W = λ @ cov_class1 @ W
        eigenvalues, eigenvectors = linalg.eigh(cov0, cov1)

        #Now, we sort by eignenvalues in desc order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        # Now, components at the begining of list are such that var for class0 is minimized while var for class1 is maximized.
        # At the end of list, we have the opposite.
        # We need to select  n_components/2 from top and bottom

        selected_components = np.concatenate(
            eigenvectors[:, :self.n_components // 2], #top
            eigenvectors[:, - self.n_components // 2:], #bottom
            axis=1
        )

        self.filters_ = selected_components # (n_channels, n_components)
        self.patterns_ = linalg.pinv(self.filters_.T) #inverse for interpretation

        return self

    def transform(self, X):
        """Applies learnt CSP filters and transform trial from (n_channels, n_times) to (n_components,)"""

        n_trials = X.shape[0]

        if self.filters_ is None:
            raise ValueError("Must call fit() before transform()!")

        X_transformed = np.zeros((n_trials, self.n_components))

        
        for i in range(n_trials):

            filtered = self.filters_.T @ X[i] #produces (n_components, n_times) shape

            #collapse time dimension by computing variance
            features = np.log(np.var(filtered, axis=1))
            X_transformed[i, :] = features
            

        return X_transformed

