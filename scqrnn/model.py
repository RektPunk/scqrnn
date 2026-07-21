import numpy as np
import torch
import torch.nn as nn
import torchpsort
from numpy.typing import NDArray


class SCQRNNModel(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_quantiles: int,
        dense_features: int,
        tau: float = 1.0,
    ):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_features, dense_features),
            nn.Sigmoid(),
            nn.Linear(dense_features, dense_features),
            nn.Sigmoid(),
            nn.Linear(dense_features, num_quantiles),
        )
        self.tau = tau

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extractor(x)
        if self.training:
            return torchpsort.soft_sort(out, tau=self.tau)
        else:
            return torch.sort(out, dim=1)[0]


class SCQRNNRegressor:
    def __init__(
        self,
        quantiles: NDArray[np.float64],
        dense_features: int = 32,
        lr: float = 0.01,
        epochs: int = 5000,
        device: str = "cpu",
        tau: float = 1.0,
    ):
        self.device = torch.device(device)
        self.quantiles = torch.tensor(
            quantiles,
            dtype=torch.float32,
            device=self.device,
        )
        self.num_quantiles = len(quantiles)
        self.dense_features = dense_features
        self.lr = lr
        self.epochs = epochs
        self.tau = tau
        self.model: SCQRNNModel | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_tensor = torch.tensor(
            y,
            dtype=torch.float32,
            device=self.device,
        ).reshape(-1, 1)

        self.model = SCQRNNModel(
            in_features=X.shape[1],
            num_quantiles=self.num_quantiles,
            dense_features=self.dense_features,
            tau=self.tau,
        ).to(self.device)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            preds = self.model(X_tensor)
            diff = y_tensor - preds
            loss = torch.maximum(
                self.quantiles * diff, (self.quantiles - 1.0) * diff
            ).mean()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 1000 == 0 or epoch == 0:
                print(f"Epoch [{epoch + 1}/{self.epochs}] Loss={loss.item():.4f}")

        return self

    @torch.no_grad()
    def predict(
        self,
        X: NDArray[np.float64],
    ) -> NDArray[np.float32]:
        if self.model is None:
            raise ValueError("Please call .fit() first.")

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)

        return self.model(X_tensor).cpu().numpy()
