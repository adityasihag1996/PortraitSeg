import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy_loss(inputs, targets)
        # The focal loss can be computed using the cross entropy loss values
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return ce_loss + focal_loss
