import numpy as np
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class HallucinationProbe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self._pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=128, random_state=42)),
            ('clf', LogisticRegression(
                penalty='l1',       
                C=0.03,             
                class_weight='balanced',
                max_iter=1000,
                solver='liblinear',
                random_state=42
            ))
        ])
        self._threshold: float = 0.5

    def _build_network(self, input_dim: int) -> None:
        pass 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        X_np = x.detach().cpu().numpy()
        probs = self._pipeline.predict_proba(X_np)[:, 1]
        return torch.from_numpy(probs).to(x.device)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        self._pipeline.fit(X, y)
        return self

    def fit_hyperparameters(self, X_val: np.ndarray, y_val: np.ndarray) -> "HallucinationProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.linspace(0.1, 0.9, 100)
        best_threshold, best_acc = 0.5, 0.0
        
        for c in candidates:
            y_c = (probs >= c).astype(int)
            score = accuracy_score(y_val, y_c)
            if score > best_acc:
                best_acc, best_threshold = score, float(c)
                
        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._pipeline.predict_proba(X)