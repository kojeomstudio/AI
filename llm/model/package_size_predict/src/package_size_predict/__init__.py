"""
Package Size Prediction - PyTorch linear model

This package provides utilities to generate sample Excel data and to train
an extremely simple linear regression model that predicts package size from
date (encoded as an ordinal day index).

Modules:
- data: Sample data generation and loading from Excel
- model: Minimal linear regression model in PyTorch
- train: Training/validation loop with console and file logging
"""

__all__ = [
    "data",
    "model",
    "train",
]

