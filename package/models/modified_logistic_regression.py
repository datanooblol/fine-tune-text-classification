"""
Modified Logistic Regression for Positive–Unlabeled (PU) Learning
=================================================================

This implementation follows the algorithm described in:

Jaskie, Elkan, Spanias — "A Modified Logistic Regression for Positive and
Unlabeled Learning"

---------------------------------------------------------------------
PROBLEM SETTING
---------------------------------------------------------------------

In Positive–Unlabeled learning we observe:

    x  -> feature vector
    s  -> label indicator

where

    s = 1   labeled positive
    s = 0   unlabeled (contains both positive and negative)

We do NOT observe the true label y.

Assumption (SCAR):

    P(s = 1 | x, y = 1) = c

Meaning:
A positive example is labeled with constant probability c.

---------------------------------------------------------------------
MODEL
---------------------------------------------------------------------

The model predicts the probability that a sample is labeled:

    g(x) = P(s = 1 | x)

Modified logistic regression:

    g(x) = 1 / (1 + b² + exp(-wᵀx))

Unlike standard logistic regression:

    σ(x) = 1 / (1 + exp(-wᵀx))

The added term b² controls the maximum probability.

Upper bound:

    max g(x) = 1 / (1 + b²)

This corresponds to the estimate of

    ĉ = P(s = 1 | y = 1)

---------------------------------------------------------------------
RECOVERING THE TRUE CLASS PROBABILITY
---------------------------------------------------------------------

From the PU relationship:

    P(s = 1 | x) = P(y = 1 | x) * c

Therefore:

    P(y = 1 | x) = P(s = 1 | x) / c

So the final classifier is:

    f(x) = g(x) / ĉ

---------------------------------------------------------------------
LOSS FUNCTION
---------------------------------------------------------------------

We model s as Bernoulli:

    s ~ Bernoulli(g(x))

Likelihood for one sample:

    P(s|x) = g(x)^s (1-g(x))^(1-s)

Log-likelihood for dataset:

    log L =
        Σ [ s log g(x) + (1-s) log(1-g(x)) ]

We minimize the negative log-likelihood:

    L =
        - mean( s log g(x) + (1-s) log(1-g(x)) )

This is identical to Binary Cross Entropy.

---------------------------------------------------------------------
ALGORITHM (PSEUDOCODE)
---------------------------------------------------------------------

Input:
    X : feature matrix
    s : labels (1 = labeled positive, 0 = unlabeled)

Initialize:
    weight vector w
    parameter b

Repeat for epochs:

    z = wᵀx

    g(x) = 1 / (1 + b² + exp(-z))

    loss =
        - mean(
            s log g(x)
            +
            (1-s) log(1-g(x))
        )

    update parameters using gradient descent

After training:

    ĉ = 1 / (1 + b²)

Prediction:

    g(x) = 1 / (1 + b² + exp(-wᵀx))

    P(y=1|x) = g(x) / ĉ

NOTE:
We prefer using `torch.nn.BCEWithLogitsLoss` instead of manually computing
binary cross entropy or using `BCELoss` after a sigmoid.

Reason:
BCEWithLogitsLoss combines a Sigmoid activation and Binary Cross Entropy
into a single operation that is numerically stable and prevents issues
like log(0), overflow, and unstable gradients.

IMPORTANT:
When using BCEWithLogitsLoss, the model should output raw logits
(NO sigmoid in the model). The loss function will apply it internally.

Example:
criterion = torch.nn.BCEWithLogitsLoss()
loss = criterion(logits, targets)
"""

# -------------------------------------------------------------------
# Standard Imports
# -------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


# -------------------------------------------------------------------
# Internal PyTorch Model
# -------------------------------------------------------------------

class _MLRModel(nn.Module):
    """
    Internal neural module implementing the modified logistic regression.

    Mathematical form:

        z = wᵀx

        g(x) = 1 / (1 + b² + exp(-z))

    Parameters learned:

        w  -> weight vector
        b  -> scalar controlling probability upper bound
    """

    def __init__(self, input_dim):
        super().__init__()

        # Linear layer computes:
        #
        #     z = wᵀx
        #
        # bias=False because the original formulation
        # only includes wᵀx
        self.linear = nn.Linear(input_dim, 1, bias=False)

        # Learnable scalar parameter b
        #
        # Used to control asymptotic probability limit
        self.b = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):

        # Linear score
        #
        #     z = wᵀx
        z = self.linear(x)

        # Modified logistic function
        #
        #     g(x) = 1 / (1 + b² + exp(-z))
        #
        # g(x) represents:
        #
        #     P(s = 1 | x)
        #
        g = 1 / (1 + self.b**2 + torch.exp(-z))

        return g


# -------------------------------------------------------------------
# Sklearn-Compatible Estimator
# -------------------------------------------------------------------

class ModifiedLogisticRegressionPU(BaseEstimator, ClassifierMixin):
    """
    Sklearn-compatible PU learning classifier.

    This estimator behaves like a standard sklearn classifier
    but internally implements the Modified Logistic Regression
    algorithm for Positive–Unlabeled learning.
    """

    def __init__(
        self,
        lr=1e-3,
        epochs=1000,
        batch_size=None,
        verbose=False,
        random_state=None,
    ):

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.random_state = random_state


    # ----------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------

    def fit(self, X, y):
        """
        Train the model.

        Inputs:
            X : feature matrix
            y : label indicator

        where

            y = 1 -> labeled positive
            y = 0 -> unlabeled
        """

        # Validate inputs
        X, y = check_X_y(X, y)

        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        # Optional reproducibility
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        n_features = X.shape[1]

        # Create internal model
        self.model_ = _MLRModel(n_features)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        # ------------------------------------------------------------
        # Training loop
        # ------------------------------------------------------------

        for epoch in range(self.epochs):

            # Forward pass
            g = self.model_(X)

            # Binary cross entropy / negative log likelihood
            #
            # L =
            #  - mean(
            #       y log g
            #       +
            #       (1-y) log(1-g)
            #    )
            #
            loss = -torch.mean(
                y * torch.log(g + 1e-10) +
                (1 - y) * torch.log(1 - g + 1e-10)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if self.verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}  Loss {loss.item():.4f}")

        # ------------------------------------------------------------
        # Estimate class prior
        # ------------------------------------------------------------

        # Learned parameter b
        b = self.model_.b.detach()

        # Estimate c
        #
        #     ĉ = 1 / (1 + b²)
        #
        self.c_hat_ = (1 / (1 + b**2)).item()

        return self


    # ----------------------------------------------------------------
    # Probability Prediction
    # ----------------------------------------------------------------

    def predict_proba(self, X):
        """
        Predict probabilities.

        Returns:

            [ P(y=0), P(y=1) ]
        """

        check_is_fitted(self, ["model_", "c_hat_"])

        X = check_array(X)

        X = torch.tensor(X, dtype=torch.float32)

        # g(x) = P(s=1|x)
        g = self.model_(X)

        # Recover P(y=1|x)
        #
        #     f(x) = g(x) / ĉ
        #
        f = g / self.c_hat_

        # Clamp for numerical safety
        f = torch.clamp(f, 0, 1)

        p = f.detach().numpy()

        # Return sklearn format
        #
        # [P(y=0), P(y=1)]
        #
        return np.hstack([1 - p, p])


    # ----------------------------------------------------------------
    # Class Prediction
    # ----------------------------------------------------------------

    def predict(self, X):
        """
        Predict class labels.

        Decision rule:

            ŷ = 1 if P(y=1|x) ≥ 0.5
        """

        proba = self.predict_proba(X)[:, 1]

        return (proba >= 0.5).astype(int)