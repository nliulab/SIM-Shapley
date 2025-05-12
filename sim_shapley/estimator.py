"""
This module is the main part to implement our SIM-Shapley method, the iterative process to update delta.
"""
import numpy as np
from scipy.linalg import inv
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from sim_shapley.utils import sample_z, sample_dataset, get_distribution, verify_model_data


class SIM_Shapley:
    def __init__(self, raw_data, imputer, loss_func, l2_penalty=0.1, t=0.5, method_type="global"):
        """
        __init__
        Args:
            raw_data: data used for sampling marginal distribution, which must have more samples than X.
            imputer: imputer class.
            loss_func: loss fuction that is callable by inputting (y_true, y_hat)
            l2_penalty: lambda value of l2_penalty.
            t: the value for updating beta when update_rule=="1".
            method_type: "global" or "local".
        """
        self.imputer = imputer
        self.loss_func = loss_func
        self.raw_data = raw_data
        self.l2_penalty = l2_penalty
        self.method_type = method_type
        self.t = t


    def __call__(self, X, Y, z_sample_num, data_batch_size=512, max_iteration_num=1e5,
                 thresh=0.05, n_jobs=-1, random_state=None, avoid_negative=False, negative_ratio=0.3,
                 verbose=False, return_convergence_process=False):
        """
        Main sampling process of SIM-Shapley.
        Args:
            X: data used for estimating shapley values, shape: (num_of_samples, feature_num).
            Y: data label or target value of X.
            z_sample_num: sampling number of z_i
            data_batch_size: data sampling number for each estimating process.
            max_iteration_num: max_iteration_num
            thresh: threshold of convergence detection
            n_jobs: number of jobs used for parallel computing. -1 means using all cpu cores.
            random_state: random_state for np.random.default_rng().
            avoid_negative：undertake stable-SIM-Shapley.
            negative_ratio: the ratio to detect negative sampling.
            verbose：verbose.
            return_convergence_process: return convergence ratio in each iteration.

        Returns:
            beta: shapley values vector.
        """
        if self.method_type == "global":
            feature_num = X.shape[1]
        else:
            feature_num = len(X)

        # verify data form and hyper-para
        # assert X.shape[0] <= self.raw_data.shape[0], "Raw data is too few."
        assert feature_num == self.raw_data.shape[1], "Data not match!"
        assert 0 < thresh < 1, "Invalid threshold!"
        # verify model data, especially for classification task.
        X, Y = verify_model_data(self.imputer, X, Y, self.loss_func, data_batch_size)

        # p(z)
        weights = get_distribution(feature_num)

        # same sample random result by rng
        self.rng = np.random.default_rng(seed=random_state)
        # initialize delta and beta
        beta = np.zeros(feature_num)
        # Record mean and std of beta
        var = beta.copy()
        mean = beta.copy()

        # Calculate v(1) and v(2)
        if self.method_type == "local":
            X = X.reshape(1, -1)
        self.v_1 = - self.loss_func(Y, self.imputer(X, np.ones(feature_num, dtype=bool)))
        self.v_0 = - self.loss_func(Y, self.imputer(X, np.zeros(feature_num, dtype=bool)))
        # print("v1:",self.v_1)
        # tqdm
        bar = tqdm(total=1)
        converge_record_list = []

        # main loop
        for i in range(int(max_iteration_num)):
            # rng makes each seed number is the same even when we restart python
            seed = self.rng.integers(low=1, high=int(max_iteration_num))
            # if i % 10 == 0:
            if verbose:
                print(f"{i}th iteration.")
            Z = sample_z(z_sample_num, feature_num, weights, seed)
            # print("Z:",Z.shape)
            if self.method_type == "global":
                X_sample, Y_sample = sample_dataset(data_batch_size, X, Y, seed)
            else:
                X_sample, Y_sample = X.reshape(1,-1), Y
                # print(X_sample.shape)
                assert len(X_sample) == 1, "Local method is not used properly! Only one sample is needed."

            # print(X_sample.shape, Y_sample.shape)
            # Calculate E[v(i)]: expected loss_value for zi
            # parallelize this for-loop.
            results = Parallel(n_jobs=n_jobs)(delayed(self._compute_zi)(X_sample, Y_sample, zi) for zi in Z)
            _, bi_list, A_list = zip(*results)

            # Calculate parameters like b and A
            b = np.mean(bi_list, axis=0)
            A = np.mean(A_list, axis=0)
            old_beta = beta.copy()
            delta, beta = self._update_delta(beta, A, b, self.v_1, self.v_0, i+1, avoid_negative)

            # update mean and std by welford Algo
            mean_new = mean + (beta - mean) / (i+1)
            var_new = var + ((beta - mean) * (beta - mean_new) - var) / (i + 1)


            # avoid negative sampling
            var_l2 = 0
            if i > 1 and avoid_negative:
                # Record var change
                var_gap = np.linalg.norm(var_new - var)
                if var_l2 == 0:
                    var_l2 = np.linalg.norm(var)
                var_ratio = var_gap / var_l2
                # print(var_gap, var_l2, var_ratio)
                if var_ratio > negative_ratio:
                    beta = old_beta
                    if verbose:
                        print('Negative sampling detected!')
                    continue


            mean = mean_new
            var = var_new
            ratio = 1

            # convergence detection
            if i > 0:
                std = np.sqrt(np.max(var)/(i+1))
                mean_gap = max(mean_new.max()-mean_new.min(), 1e-12)
                ratio = std / mean_gap
                if return_convergence_process:
                    converge_record_list.append(ratio)
                if verbose:
                    print('ratio:', ratio)

            if ratio < thresh:
                print(f"Convergence detected! Gap is {ratio}.")
                bar.n = bar.total
                bar.refresh()
                break

            # Update using convergence estimation.
            N_est = (i + 1) * (ratio / thresh) ** 2
            bar.n = np.around((i + 1) / N_est, 4)
            bar.refresh()

        bar.close()

        if return_convergence_process:
            return beta, converge_record_list
        else:
            return beta


    def _update_delta(self, beta, A, b, v_1, v_0, n, avoid_negative=False):
        """
        Utilize our explicit solution on new delta with A and b.
        Args:
            beta: beta in previous iteration.
            A: A.
            b: b.
            v_1: v(1).
            v_0: v(0).
            n: current iteration number.

        Returns:
            delta: P by 1 array, delta derived in this iteration.
            new_beta: new_beta in current iteration.
        """
        vec_1 = np.ones(len(b)) # (feature_num, )

        assert 0 < self.t < 1, "t must between o and 1!"
        A_bar = A + np.eye(A.shape[0]) * self.l2_penalty
        A_bar_inv = inv(A_bar)  # must be invertible
        if avoid_negative and n < 5:
            c1 = self.t / (1 - self.t**n)
            c2 = (1-self.t) / (1 - self.t**n)
            gap = b - A @ (beta * c1)
            delta = A_bar_inv @ (gap + vec_1 * ((v_1 - v_0 - vec_1.T @ (beta * c1) - vec_1.T @ A_bar_inv @ gap) /
                                                (vec_1.T @ A_bar_inv @ vec_1))) / c2
            new_beta = beta * c1 + delta * c2
        else:
            gap = b - A @ (beta * self.t)
            delta = A_bar_inv @ (gap + vec_1 * ((v_1 - v_0 - vec_1.T @ (beta * self.t) - vec_1.T @ A_bar_inv @ gap) /
                                                (vec_1.T @ A_bar_inv @ vec_1))) / (1 - self.t)
            new_beta = beta * self.t + delta * (1 - self.t)

        return delta, new_beta


    def _compute_zi(self, X_sample, Y_sample, zi):
        """
        Compute A and b for zi.
        Args:
            X_sample: X_sample
            Y_sample: Y_sample
            zi: zi

        Returns:
        loss_value, bi, A_i
        """
        y_hat = self.imputer(X_sample, zi)
        loss_value =  - self.loss_func(Y_sample, y_hat)
        bi = zi * (loss_value - self.v_0)
        A_i = np.outer(zi, zi).astype(int)

        return loss_value, bi, A_i