"""
This module is to build utility functions for our project.
"""
import sys
import numpy as np


def get_distribution(feature_num):
    """
    Get p(z).
    Notice: We don't sample the situation where z concludes every element as well as nothing, because the least square term will be zero
    in this cass bringing out unnecessary computation.
    Args:
        feature_num: feature_num

    Returns:
        weights: (feature_num-1,)
    """
    weights = np.arange(1, feature_num)
    weights = 1 / (weights * (feature_num - weights))
    weights = weights / np.sum(weights)  # Distribution of each size of subset.
    return weights

def sample_z(sample_num, feature_num, weights, seed):
    """
    This func is to sample z (subset) of p(z).

    Args:
        sample_num: number of sampling z, which is the "m" in Eq.9 of the paper.
        feature_num: number of features.
        weights: p(z).
        seed: random seed

    Returns:
    sample_num by feature_num matrix of all possible z_i.
    """
    rng = np.random.default_rng(seed=seed)
    num_included = rng.choice(feature_num - 1, size=sample_num, p=weights) + 1
    S = np.zeros((sample_num, feature_num), dtype=bool)
    for row, num in zip(S, num_included):
        inds = rng.choice(feature_num, size=num, replace=False)
        row[inds] = 1

    return S


def sample_dataset(batch_size, X, Y, seed):
    """
    Sample_dataset with replacement.
    Args:
        batch_size: batch_size
        X: data
        Y: label
        seed: random seed

    Returns:
    Sampled X, Y.
    """
    rng = np.random.default_rng(seed=seed)
    indices = rng.choice(len(X), size=batch_size, replace=True)
    return X[indices], Y[indices]

def model_conversion(model):
    """Convert model to callable."""
    if safe_isinstance(model, "sklearn.base.ClassifierMixin"):
        return lambda x: model.predict_proba(x)

    elif safe_isinstance(model, "sklearn.base.RegressorMixin"):
        return lambda x: model.predict(x)

    elif safe_isinstance(model, "catboost.CatBoostClassifier"):
        return lambda x: model.predict_proba(x)

    elif safe_isinstance(model, "catboost.CatBoostRegressor"):
        return lambda x: model.predict(x)

    elif safe_isinstance(model, "lightgbm.basic.Booster"):
        return lambda x: model.predict(x)

    elif safe_isinstance(model, "xgboost.core.Booster"):
        import xgboost

        return lambda x: model.predict(xgboost.DMatrix(x))

    elif safe_isinstance(model, "torch.nn.Module"):
        print(
            "Setting up imputer for PyTorch model, assuming that any "
            "necessary output activations are applied properly. If "
            "not, please set up nn.Sequential with nn.Sigmoid or nn.Softmax"
        )

        import torch

        model.eval()
        device = next(model.parameters()).device
        return (
            lambda x: model(torch.tensor(x, dtype=torch.float32, device=device))
            .cpu()
            .data.numpy()
        )

    elif safe_isinstance(model, "keras.Model"):
        print(
            "Setting up imputer for keras model, assuming that any "
            "necessary output activations are applied properly. If not, "
            "please set up keras.Sequential with keras.layers.Softmax()"
        )

        return lambda x: model(x, training=False).numpy()

    elif callable(model):
        # Assume model is compatible function or callable object.
        return model

    else:
        raise ValueError(
            "model cannot be converted automatically, "
            "please convert to a lambda function"
        )

def safe_isinstance(obj, class_str):
    """Check isinstance without requiring imports."""
    if not isinstance(class_str, str):
        return False
    module_name, class_name = class_str.rsplit(".", 1)
    if module_name not in sys.modules:
        return False
    module = sys.modules[module_name]
    class_type = getattr(module, class_name, None)
    if class_type is None:
        return False
    return isinstance(obj, class_type)


def crossentropyloss(target, pred, reduction="mean"):
    '''Cross entropy loss that does not average across samples.'''
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
        pred = np.concatenate((1 - pred, pred), axis=1)

    if pred.shape == target.shape:
        # Soft cross entropy loss.
        pred = np.clip(pred, a_min=1e-12, a_max=1-1e-12)
        loss =  - np.sum(np.log(pred) * target, axis=1)
    else:
        # Standard cross entropy loss.
        loss = - np.log(pred[np.arange(len(pred)), target])

    if reduction == "mean":
        return np.mean(loss)
    else:
        return loss



def mseloss(target, pred):
    '''MSE loss that does not average across samples.'''
    if len(pred.shape) == 1:
        pred = pred[:, np.newaxis]
    if len(target.shape) == 1:
        target = target[:, np.newaxis]
    return np.sum((pred - target) ** 2, axis=1)


def zero_one_loss(target, pred, reduction="mean"):
    """zero-one loss that expects probabilities."""
    if pred.ndim == 1:
        pred = pred[:, np.newaxis]
    if pred.shape[1] == 1:
        pred = np.append(1 - pred, pred, axis=1)

    if target.ndim == 1:
        # Class labels.
        loss = (np.argmax(pred, axis=1) != target).astype(float)
    elif target.ndim == 2:
        # Probabilistic labels.
        loss = (np.argmax(pred, axis=1) != np.argmax(target, axis=1)).astype(float)
    else:
        raise ValueError("incorrect labels shape for zero-one loss")

    if reduction == "mean":
        return np.mean(loss)
    return loss


def dataset_output(imputer, X, batch_size):
    """Get model output for entire dataset."""
    Y = []
    for i in range(int(np.ceil(len(X) / batch_size))):
        x = X[i * batch_size : (i + 1) * batch_size]
        pred = imputer(x, np.ones((len(x), imputer.num_groups), dtype=bool))
        Y.append(pred)
    return np.concatenate(Y)


def verify_model_data(imputer, X, Y, loss, batch_size):
    """Ensure that model and data are set up properly."""
    check_labels = True
    if Y is None:
        print("Calculating model sensitivity (Shapley Effects, not SAGE)")
        check_labels = False
        Y = dataset_output(imputer, X, batch_size)

        # Fix output shape for classification tasks.
        if safe_isinstance(loss, ("crossentropyloss", "zero_one_loss")):
            if Y.shape == (len(X),):
                Y = Y[:, np.newaxis]
            if Y.shape[1] == 1:
                Y = np.concatenate([1 - Y, Y], axis=1)

    if safe_isinstance(loss, ("crossentropyloss", "zero_one_loss")):
        x = X[:batch_size]
        probs = imputer(x, np.ones((len(x), imputer.num_groups), dtype=bool))

        # Check labels shape.
        if check_labels:
            Y = Y.astype(int)
            if Y.shape == (len(X),):
                # This is the preferred shape.
                pass
            elif Y.shape == (len(X), 1):
                Y = Y[:, 0]
            else:
                raise ValueError(
                    "labels shape should be (batch,) or (batch, 1)"
                    " for cross entropy loss"
                )

        if (probs.ndim == 1) or (probs.shape[1] == 1):
            # Check label encoding.
            if check_labels:
                unique_labels = np.unique(Y)
                if np.array_equal(unique_labels, np.array([0, 1])):
                    # This is the preferred labeling.
                    pass
                elif np.array_equal(unique_labels, np.array([-1, 1])):
                    # Set -1 to 0.
                    Y = Y.copy()
                    Y[Y == -1] = 0
                else:
                    raise ValueError(
                        "labels for binary classification must be [0, 1] or [-1, 1]"
                    )

            # Check for valid probability outputs.
            valid_probs = np.all(np.logical_and(probs >= 0, probs <= 1))

        elif probs.ndim == 2:
            # Multiclass output, check for valid probability outputs.
            valid_probs = np.all(np.logical_and(probs >= 0, probs <= 1))
            ones = np.sum(probs, axis=1)
            valid_probs = valid_probs and np.allclose(ones, np.ones(ones.shape))

        else:
            raise ValueError("prediction has too many dimensions")

        if not valid_probs:
            raise ValueError("predictions are not valid probabilities")

    return X, Y