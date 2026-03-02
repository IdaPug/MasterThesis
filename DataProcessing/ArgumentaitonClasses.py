import torch


class GaussianNoiseMedical:
    def __init__(self, sigma=0.01, p=0.3):
        self.sigma = sigma
        self.p = p

    def __call__(self, image, mask):
        if torch.rand(1) < self.p:
            noise = torch.randn_like(image) * self.sigma
            image = image + noise
        return image, mask


class RandomGamma:
    def __init__(self, gamma_range=(0.9, 1.1), p=0.3, eps=1e-6):
        self.gamma_range = gamma_range
        self.p = p
        self.eps = eps

    def __call__(self, image, mask):
        if torch.rand(1) < self.p:
            gamma = torch.empty(1).uniform_(*self.gamma_range).item()
            image = torch.sign(image) * torch.pow(torch.abs(image) + self.eps, gamma)
        return image, mask
