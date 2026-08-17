import torch.nn as nn


class BatteryFailureNet(nn.Module):
    """
    Same architecture used in training (Cell 9 / Cell 20 of the notebook).
    hidden_sizes and dropout are read from artifacts/feature_config.json
    so this stays in sync with whatever configuration was trained.
    """

    def __init__(self, input_dim, hidden_sizes=(128, 64, 32), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
