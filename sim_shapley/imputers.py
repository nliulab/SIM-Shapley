"""
This module is the imputer to undertake marginal distribution sampling and shapley value computation (loss function value)
process.
"""
import warnings
import numpy as np
from sim_shapley import utils

class Imputer:
    """
    Imputer base class. This class is used to sample marginal distribution of dataset and calculate expected shapley value
    for specific subset: zi.
    """

    def __init__(self, model, raw_data):
        self.model = utils.model_conversion(model)
        self.raw_data = raw_data

    def __call__(self, x, S):
        raise NotImplementedError

class MarginalImputer(Imputer):
    """Marginalizing out removed features with their marginal distribution."""

    def __init__(self, model, raw_data, sample_num = 10):
        super().__init__(model, raw_data)
        self.data = raw_data
        self.data_repeat = raw_data.copy()
        self.sample_num = sample_num
        self.feature_num = raw_data.shape[1]


        if len(raw_data) > 1024:
            warnings.warn(
                f"using {len(raw_data)} background samples may lead to slow "
                "runtime, consider using <= 1024",
                RuntimeWarning,
            )

    def __call__(self, x, S):
        # Prepare x and S.
        # S is z_i of our paper.
        # each x_i repeat 10 times.
        x_repeat = x.repeat(self.sample_num, 0)
        # mask matrix
        n = len(x_repeat)
        S = np.tile(S, (n, 1))

        # Prepare marginal samples.
        if len(self.data_repeat) != n:
            repeat_time = n // len(self.data) + 1
            self.data_repeat = np.tile(self.data, (repeat_time, 1))[:n]

        # print(S.shape, self.data_repeat.shape)
        x_repeat[~S] = self.data_repeat[~S]
        # Make predictions.
        pred = self.model(x_repeat)
        pred = pred.reshape(-1, self.sample_num, *pred.shape[1:])
        # print(np.mean(pred, axis=1).shape)
        return np.mean(pred, axis=1)