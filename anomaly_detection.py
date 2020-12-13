import numpy as np
import torch
from sklearn.base import BaseEstimator

import utils.dataset
import utils.model
from mcr import MaximalCodingRateReduction


class AnomalyDetection(BaseEstimator):
    def __init__(self, alpha: float = 0.05):
        super(BaseEstimator, self).__init__()
        self.alpha = alpha

    def fit(self, Z: torch.tensor, Y: torch.tensor, eps: float = 0.01) -> None:
        """
        :param Z: torch.tensor(n_samples, n_nn_gen_features), featurized data matrix
        :param Y: torch.tensor(n_samples, ), vector of labels
        :param eps: epsilon^2 for MCR method.
        """
        self.score = self._get_score_function(Z, Y, eps)
        scores = [self.score(Z[i]) for i in range(Z.shape[0])]
        self.score_cutoff = np.quantile(scores, 1 - self.alpha)

    def _get_score_function(self, Z: torch.tensor, Y: torch.tensor, eps: float = 0.01):
        """
        Returns a modified Mahalobinis score function to use in anomaly detection.
        :param Z: torch.tensor(n_samples, n_nn_gen_features), featurized data matrix
        :param Y: torch.tensor(n_samples, ), class membership vector
        :param eps: epsilon^2 for MCR method.
        :return: a score function taking in a vector (n_nn_gen_features, ) and outputting a score.
        """
        m, p = Z.shape[0], Z.shape[1]
        num_classes = len(Y.unique())
        Pi = utils.dataset.label_to_membership(Y, num_classes)
        Sigma = torch.tensor(np.zeros(shape=(Pi.shape[0], p, p))).cuda()
        tr_Pi = torch.tensor(np.zeros(shape=(Pi.shape[0],))).cuda()
        mu = torch.tensor(np.zeros(shape=(Pi.shape[0], p))).cuda()
        for j in range(Pi.shape[0]):
            tr_Pi[j] = torch.sum(Pi[j]) + MaximalCodingRateReduction.INV_EPS
            Sigma[j] = Z.T.matmul(Z * Pi[j][:, None]) / tr_Pi[j] # hack to save memory https://stackoverflow.com/questions/53987906/how-to-multiply-row-wise-by-scalar-in-pytorch
            mu[j] = torch.mean(Z[Y == j])

        I = torch.eye(p).cuda()
        mahalobinis_inverse_covariances = torch.inverse((eps / p) * I + Sigma).cuda()
        def score_fn(z: torch.tensor) -> float:
            """
            A score function based on Mahalobinis distance.
            :param z: torch.tensor(n_nn_gen_features, ), sample to score
            :return: the score of z
            """
            centered_z = z - mu
            return min(
                centered_z[j].T.matmul(mahalobinis_inverse_covariances[j]).matmul(centered_z[j]) + (2 * eps / p) * torch.log(tr_Pi[j] / m)
                for j in range(Pi.shape[0])
            )
        return score_fn

    def predict(self, Z):
        predictions = torch.zeros((Z.shape[0],)).cuda()
        for j in range(Z.shape[0]):
            predictions[j] = int(self.score(Z[j]) > self.score_cutoff)
        return predictions
