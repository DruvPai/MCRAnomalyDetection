import utils.dataset
import torch


class MaximalCodingRateReduction(torch.nn.Module):
    INV_EPS = 1e-8  # ensures invertibility

    def __init__(self, eps: float = 0.01, gamma_1: float = 1, gamma_2: float = 1) -> None:
        super(MaximalCodingRateReduction, self).__init__()
        self.eps = eps
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2

    def discriminative_loss(self, Z: torch.tensor, gamma: float = 1) -> torch.tensor:
        m, p = Z.shape
        I = torch.eye(p).cuda()
        return torch.logdet(I + gamma * p / (m * self.eps) * Z.T.matmul(Z)) / 2

    def empirical_discriminative_loss(self, Z: torch.tensor) -> torch.tensor:
        return self.discriminative_loss(Z, self.gamma_1)

    def theoretical_discriminative_loss(self, Z: torch.tensor) -> torch.tensor:
        return self.discriminative_loss(Z, 1.)

    def compressive_loss(self, Z: torch.tensor, Pi: torch.tensor) -> torch.tensor:
        m, p = Z.shape
        k, _ = Pi.shape
        I = torch.eye(p).cuda()
        compressive_loss = 0.
        for j in range(k):
            tr_Pi_j = torch.sum(Pi[j]) + self.INV_EPS
            log_det = torch.logdet(I + p / (tr_Pi_j * self.eps) * Z.T.matmul(Z * Pi[j][:, None])) # hack to save memory https://stackoverflow.com/questions/53987906/how-to-multiply-row-wise-by-scalar-in-pytorch
            compressive_loss += log_det * tr_Pi_j / (2 * m)
        return compressive_loss

    def forward(self, Z, Y, num_classes=None):
        if num_classes is None:
            num_classes = len(Y.unique())
        Pi = utils.dataset.label_to_membership(Y, num_classes)

        emp_disc_loss = self.empirical_discriminative_loss(Z)
        theo_disc_loss = self.theoretical_discriminative_loss(Z)
        comp_loss = self.compressive_loss(Z, Pi)

        emp_total_loss = comp_loss - self.gamma_2 * comp_loss
        return (emp_total_loss,
                [emp_disc_loss.item(), comp_loss.item()],
                [theo_disc_loss.item(), comp_loss.item()]
                )

