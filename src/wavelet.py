from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class HaarWaveletTransform(BaseEstimator, TransformerMixin):
    """Apply Haar to each channel, keep 3D shape for CSP."""
    def __init__(self, n_levels=4):
        self.n_levels = n_levels
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """
        X: (n_trials, n_channels, n_times)
        Returns: (n_trials, n_channels, n_wavelet_coeffs)
        """
        n_trials, n_channels, n_times = X.shape
        
        # Apply DWT to get coefficient length
        sample_coeffs = self.haar_dwt_multilevel(X[0, 0, :], n_levels=self.n_levels)
        n_coeffs = sum(len(c) for c in sample_coeffs)
        
        X_wavelet = np.zeros((n_trials, n_channels, n_coeffs))
        
        for trial in range(n_trials):
            for ch in range(n_channels):
                coeffs = self.haar_dwt_multilevel(X[trial, ch, :], n_levels=self.n_levels)
                # Flatten coefficients
                X_wavelet[trial, ch, :] = np.concatenate(coeffs)
        
        # Replace any inf/nan
        X_wavelet = np.nan_to_num(X_wavelet, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return X_wavelet

    def haar_dwt_forward(self, signal):
        """One-level DWT - vectorized"""
        N = len(signal)
        if N % 2:
            signal = np.append(signal, signal[-1])
            N = len(signal)
        
        # Reshape to pairs: (N//2, 2)
        pairs = signal.reshape(N//2, 2)
        
        # Vectorized operations
        approx = (pairs[:, 0] + pairs[:, 1]) / np.sqrt(2)
        detail = (pairs[:, 0] - pairs[:, 1]) / np.sqrt(2)
        
        return approx, detail

    def haar_dwt_multilevel(self, signal, n_levels=5):
        """
        Multi-level Haar DWT - vectorized.
        
        Returns: [approximation, detail_n, detail_n-1, ..., detail_1]
        """
        coeffs = []
        current = signal.copy()
        
        for _ in range(n_levels):
            if len(current) < 2:
                break
            approx, detail = self.haar_dwt_forward(current)
            coeffs.append(detail)
            current = approx
        
        coeffs.append(current)
        return coeffs[::-1]