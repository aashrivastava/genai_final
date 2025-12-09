from abc import ABC, abstractmethod
import torch
from sklearn.linear_model import LogisticRegression
import numpy as np


class BaseProbe(ABC):
    @abstractmethod
    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        """Predicts logits for activations

        Args:
            activations (torch.Tensor): n x d tensor of activations

        Returns:
            torch.Tensor: n x 1 tensor of logits
        """
        pass

    @abstractmethod
    def predict_proba(self, activations: torch.Tensor) -> torch.Tensor:
        """Predicts probabilities for activations

        Args:
            activations (torch.Tensor): n x d tensor of activations

        Returns:
            torch.Tensor: n x 1 tensor of probabilities
        """
        pass


class MassMeanProbe(BaseProbe):
    direction: torch.Tensor

    def __init__(self, direction: torch.Tensor):
        self.direction = direction

    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        return self.direction @ activations.T

    def predict_proba(self, activations: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.predict(activations))


class LogisticRegressionProbe(BaseProbe):
    def __init__(self, activations: torch.Tensor, labels: list[bool]):
        self.lr = LogisticRegression()
        self.lr.fit(activations.cpu().numpy(), np.array(labels))

    def predict(self, activations: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.lr.predict(activations.cpu().numpy()))

    def predict_proba(self, activations: torch.Tensor) -> torch.Tensor:
        # Get probabilities and return only the positive class probabilities
        probs = self.lr.predict_proba(activations.cpu().numpy())
        return torch.tensor(probs[:, 1])  # Return only positive class probabilities
