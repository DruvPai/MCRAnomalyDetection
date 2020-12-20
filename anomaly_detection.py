import numpy as np
import torch
from sklearn.base import BaseEstimator


class AnomalyDetection(BaseEstimator):
    def __init__(self):
        super(BaseEstimator, self).__init__()

    def fit(self, Z: torch.tensor, Y: torch.tensor, eps: float = 0.01) -> None:
        pass

    def predict(self, Z: torch.tensor) -> torch.tensor:
        pass

    def fit_predict(self, Z: torch.tensor, Y: torch.tensor, eps: float = 0.01) -> torch.tensor:
        self.fit(Z, Y, eps)
        return self.predict(Z)


class AnomalyDetectionMahalanobis(AnomalyDetection):

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

        identity = torch.eye(p).cuda()
        for j in range(k):
            Zj = Z[Y == j]
            mj = Zj.shape[0]
            self.means[j] = torch.mean(Zj, dim=0)
            centered = Zj - self.means[j]
            cov = centered.T.matmul(centered) / mj
            self.inv_covs[j] = torch.inverse(identity * eps / p + cov)

    def predict(self, Z: torch.tensor) -> torch.tensor:
        m, p = Z.shape
        k = self.k
        preds = torch.zeros(m, k).cuda()
        for j in range(k):
            centered = Z - self.means[j]
            preds[:, j] = torch.sum(centered * centered.matmul(self.inv_covs[j]), dim=1)

        scores, _ = torch.min(preds, dim=1)
        return scores

class AnomalyDetectionMICL(AnomalyDetection):

    def fit(self, Z: torch.tensor, Y: torch.tensor, eps: float = 0.01) -> torch.tensor:
        m, p = Z.shape
        k = len(Y.unique())

        self.k = k
        self.train_data = Z
        self.train_labels = Y
        self.eps = eps
        self.L_eps = torch.zeros(k).cuda()

        for j in range(k):
            idx = Y == j
            self.L_eps[j] = self._L(Z[idx]) + torch.log2(torch.sum(idx) / m)

    def predict(self, Z: torch.tensor) -> torch.tensor:
        m, p = Z.shape
        k = self.k
        preds = torch.zeros(m, k).cuda()
        preds -= self.L_eps
        for i in range(m):
            for j in range(k):
                preds[i][j] += self._L(torch.cat((self.train_data[self.train_labels == j], Z[i][None, :])))
        return preds

    def _L(self, Z: torch.tensor) -> torch.tensor:
        m, n = Z.shape
        mu = torch.mean(Z, dim=0)
        centered = Z - mu
        sigma = centered.T.matmul(centered) / (m - 1)
        identity = torch.eye(n).cuda()
        return ((m + n) / 2) * torch.logdet(identity + (n/self.eps) * sigma) / np.log(2) + (n/2) * torch.log2(1 + mu.matmul(mu) / self.eps)

class AnomalyDetectionAsymptoticMICL(AnomalyDetection):

    def fit(self, Z: torch.tensor, Y: torch.tensor, eps: float = 0.01) -> None:
        """
        :param Z: torch.tensor(n_samples, n_nn_gen_features), featurized data matrix
        :param Y: torch.tensor(n_samples, ), vector of labels
        :param eps: epsilon^2 for MCR method.
        """
        m, p = Z.shape
        k = len(Y.unique())
        self.m = m
        self.k = k
        self.mu = torch.zeros(k, p).cuda()
        self.mj = torch.zeros(k).cuda()
        self.inv_covs = torch.zeros(k, p, p).cuda()
        self.eff_dim = torch.zeros(k).cuda()

        identity = torch.eye(p).cuda()
        for j in range(k):
            Zj = Z[Y == j]
            self.mj[j] = Zj.shape[0]
            self.mu[j] = torch.mean(Zj, dim=0)
            centered = Zj - self.mu[j]
            sigma = centered.T.matmul(centered) / (self.mj[j] - 1)
            self.inv_covs[j] = torch.inverse(sigma + (eps / p) * identity)
            self.eff_dim[j] = torch.trace(sigma.matmul(self.inv_covs[j]))

    def predict(self, Z: torch.tensor) -> torch.tensor:
        m, p = Z.shape
        k = self.k
        preds = torch.zeros(m, k).cuda()
        preds += torch.log(self.mj / self.m) + self.eff_dim / 2
        for j in range(k):
            centered = Z - self.mu[j]
            preds[:, j] = torch.sum(centered * centered.matmul(self.inv_covs[j]), dim=1)
        scores, _ = torch.min(preds, dim=1)
        return scores

class AnomalyDetectionSubspace(AnomalyDetection):

    def fit(self, Z: torch.tensor, Y: torch.tensor, eps: float = 0.01) -> None:
        m, p = Z.shape
        k = len(Y.unique())
        self.m = m
        self.k = k
        self.subspaces = []
        for j in range(k):
            Zj = Z[Y == j]
            A = Zj.T.matmul(Zj)
            rank = torch.matrix_rank(A, symmetric=True)
            eigval, eigvec = torch.symeig(A, eigenvectors=True)
            subspace = eigvec[-rank:].T # sorted in ascending order of eigenvalues
            self.subspaces.append(subspace)

    def predict(self, Z: torch.tensor) -> torch.tensor:
        Zt = Z.T
        m, p = Z.shape
        k = self.k
        preds = torch.zeros(m, k)
        for j in range(k):
            subspace = self.subspaces[j]
            p, dim = subspace.shape
            proj, _ = torch.lstsq(Zt, subspace)
            preds[:, j] = torch.sum(torch.square(proj[dim:]), dim=0)
            # see https://pytorch.org/docs/stable/generated/torch.lstsq.html the last m - n rows hold residuals
        scores, _ = torch.min(preds, dim=1)
        return scores