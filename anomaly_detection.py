import numpy as np
import torch
from sklearn.base import BaseEstimator

import utils.dataset
import utils.model
from mcr import MaximalCodingRateReduction


class AnomalyDetection(BaseEstimator):
    def __init__(self):
        super(BaseEstimator, self).__init__()

    def fit(self, Z: torch.tensor, Y: torch.tensor, eps: float = 0.01) -> None:
        """
        :param Z: torch.tensor(n_samples, n_nn_gen_features), featurized data matrix
        :param Y: torch.tensor(n_samples, ), vector of labels
        :param eps: epsilon^2 for MCR method.
        """
        m, p = Z.shape
        k = len(Y.unique())
        self.k = k
        self.means = torch.zeros(k, p).cuda()
        self.inv_covs = torch.zeros(k, p, p).cuda()
        self.scores = torch.zeros(m).cuda()

        preds = torch.zeros(m, k).cuda()

        identity = torch.eye(p).cuda()
        for j in range(k):
            Zj = Z[Y == j]
            mj = Zj.shape[0]
            self.means[j] = torch.mean(Zj, dim=0)
            centered = Zj - self.means[j]
            cov = centered.T.matmul(centered) / mj
            self.inv_covs[j] = torch.inverse(identity * eps / p + cov)
        
        for j in range(k):
            centered = Z - self.means[j]
            preds[:, j] = torch.sum(centered * centered.matmul(self.inv_covs[j]), dim=1)

        self.scores, y_pred = torch.min(preds, dim=1)
        print("mahalanobis train accuracy: %f" % torch.mean((y_pred == Y).float()).item())

    def predict(self, Z):
        n = Z.shape[0]
        k = self.k
        preds = torch.zeros(n, k).cuda()
        for j in range(k):
            centered = Z - self.means[j]
            preds[:, j] = torch.sum(centered * centered.matmul(self.inv_covs[j]), dim=1)
        
        scores, _ = torch.min(preds, dim=1)
        return scores
