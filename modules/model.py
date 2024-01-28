import torch.nn as nn


class Model(nn.Module):
    def __init__(self, c_weight, ns, c_bits):
        """
        Deep neural network of the Generator attack. Creates an image of size 200x200 pixels for a challenge bitstring
        of arbitrary length.
        :param c_weight: number of input dimensions the challenge will be transformed into [hparam]
        :param ns: multiplier for each layer for the number of filters of the network [hparam]
        :param c_bits: number of bits in the challenge
        """
        super().__init__()
        # starting dimensions of the input for the first transposed convolution
        # -> Input will be of shape (batch_size, challenge_weight, input_dim, input_dim)
        self.init_dim = 45
        self.challenge_bits = c_bits
        self.challenge_weight = c_weight

        self.challenge_transform = nn.Linear(c_bits, self.init_dim ** 2 * c_weight)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(c_weight, ns * 8, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ns * 8),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 8, ns * 4, 6, 1, 0, bias=False),
            nn.BatchNorm2d(ns * 4),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 4, ns * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ns * 2),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 2, ns * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ns * 2),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 2, ns * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ns * 2),
            nn.GELU(),

            nn.ConvTranspose2d(ns * 2, ns, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ns),
            nn.GELU(),

            nn.ConvTranspose2d(ns, 1, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, challenge):
        bs = challenge.shape[0]
        challenge = challenge.view(bs, -1)
        challenge = self.challenge_transform(challenge)
        challenge = challenge.view(-1, self.challenge_weight, self.init_dim, self.init_dim)
        return self.main(challenge)
