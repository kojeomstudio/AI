"""Basic PyTorch machine learning samples.

This module provides simple examples for linear regression and binary
classification using PyTorch. The intent is educational and the code
is written with clarity in mind so that it can be easily maintained or
extended.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def run_linear_regression(num_epochs: int = 500) -> nn.Module:
    """Train a one layer linear regression model.

    Args:
        num_epochs: Number of training iterations.

    Returns:
        Trained PyTorch model.
    """
    # Generate a simple linear relation: y = 3x + 2 + noise
    x = torch.linspace(0, 1, 50).unsqueeze(1)
    y = 3 * x + 2 + torch.randn_like(x) * 0.1

    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

    print(f"Linear Regression final loss: {loss.item():.4f}")
    return model


def run_binary_classification(num_epochs: int = 1000) -> nn.Module:
    """Train a simple logistic regression classifier.

    A synthetic dataset with two informative features is generated using
    ``sklearn.make_classification``.

    Args:
        num_epochs: Number of training iterations.

    Returns:
        Trained PyTorch model.
    """
    X, y = make_classification(
        n_samples=200,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42,
    )

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_pred = model(X_test)
        predicted = (test_pred > 0.5).float()
        accuracy = (predicted.eq(y_test).sum() / y_test.size(0)).item()

    print(f"Classification accuracy: {accuracy:.2f}")
    return model


def run_multifeature_linear_regression(num_epochs: int = 700) -> nn.Module:
    """Train linear regression using multiple input features."""
    # Create a dataset with three input features.
    torch.manual_seed(42)
    x = torch.randn(100, 3)
    weights = torch.tensor([[1.5], [-2.0], [3.0]])
    bias = 0.5
    y = x @ weights + bias + torch.randn(100, 1) * 0.1

    model = nn.Linear(3, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

    print(f"Multi-feature Linear Regression final loss: {loss.item():.4f}")
    return model


def run_polynomial_regression(num_epochs: int = 800) -> nn.Module:
    """Fit a second degree polynomial using linear regression."""
    torch.manual_seed(0)
    x = torch.linspace(-1, 1, 80).unsqueeze(1)
    y = 5 * x ** 2 - 3 * x + 1 + torch.randn_like(x) * 0.1

    # Expand to polynomial features [x, x^2]
    poly_x = torch.cat([x, x ** 2], dim=1)

    model = nn.Linear(2, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.05)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        pred = model(poly_x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

    print(f"Polynomial Regression final loss: {loss.item():.4f}")
    return model


def run_multiclass_classification(num_epochs: int = 1000) -> nn.Module:
    """Train a softmax classifier on a 3-class toy dataset."""
    X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=7)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    model = nn.Linear(2, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for _ in range(num_epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_pred = model(X_test)
        predicted = test_pred.argmax(dim=1)
        accuracy = (predicted.eq(y_test).sum() / y_test.size(0)).item()

    print(f"Multiclass accuracy: {accuracy:.2f}")
    return model


if __name__ == "__main__":
    run_linear_regression()
    run_multifeature_linear_regression()
    run_polynomial_regression()
    run_binary_classification()
    run_multiclass_classification()

