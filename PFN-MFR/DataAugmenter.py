import numpy as np
import pandas as pd



class DataAugmenter:
    def __init__(self, method="mixup", alpha=0.2, mask_ratio=0.1, continuous_columns=None, categorical_columns=None):
        self.method = method
        self.alpha = alpha
        self.mask_ratio = mask_ratio
        self.continuous_columns = continuous_columns
        self.categorical_columns = categorical_columns


    def mixup(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        lam = np.random.beta(self.alpha, self.alpha)
        indices = np.random.permutation(X.shape[0])
        X_augmented = lam * X + (1 - lam) * X[indices]
        y_augmented = lam * y + (1 - lam) * y[indices]
        return X_augmented, y_augmented

    def hidden_mix(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        lam = np.random.beta(self.alpha, self.alpha)
        indices = np.random.permutation(X.shape[0])
        mask = np.random.rand(X.shape[1]) < lam
        X_augmented = X.copy()
        X_augmented[:, mask] = X[indices][:, mask]
        y_augmented = lam * y + (1 - lam) * y[indices]
        return X_augmented, y_augmented

    def mask_token(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        mask = np.random.rand(*X.shape) < self.mask_ratio
        X_augmented = X.copy()
        X_augmented[mask] = np.random.randn(*X_augmented[mask].shape)
        return X_augmented

    def vime_augment(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        mask = np.random.rand(*X.shape) < self.mask_ratio
        no, dim = X.shape
        X_bar = np.zeros_like(X)
        for i in range(dim):
            idx = np.random.permutation(no)
            X_bar[:, i] = X[idx, i]
        X_augmented = X * (1 - mask) + X_bar * mask
        return X_augmented



    def augment(self, X, y=None):
        if self.method == "mixup":
            return self.mixup(X, y)
        elif self.method == "hidden_mix":
            return self.hidden_mix(X, y)
        elif self.method == "mask_token":
            return self.mask_token(X), y
        elif self.method == "vime":
            return self.vime_augment(X), y
        else:
            raise ValueError(f"Unknown method: {self.method}")


