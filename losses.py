import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    """
    Multi-class Dice + Cross Entropy Loss
    - Works for both 2D (B, C, H, W) and 3D (B, C, H, W, D)
    - Excludes background (class 0) from Dice loss
    """
    def __init__(self, weight_ce=0.5, smooth=1e-5):
        super(DiceCELoss, self).__init__()
        self.weight_ce = weight_ce
        self.smooth = smooth
        self.ce_loss = nn.CrossEntropyLoss()  # can add class weights if needed

    def dice_loss(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets depending on spatial dimensionality
        if logits.dim() == 4:  # 2D: [B, C, H, W]
            targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
            dims = (0, 2, 3)
        elif logits.dim() == 5:  # 3D: [B, C, H, W, D]
            targets_one_hot = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()
            dims = (0, 2, 3, 4)
        else:
            raise ValueError(f"Unsupported input shape {logits.shape}")

        intersection = torch.sum(probs * targets_one_hot, dims)
        denominator = torch.sum(probs + targets_one_hot, dims)
        dice_per_class = (2. * intersection + self.smooth) / (denominator + self.smooth)

        # Exclude background (index 0)
        dice_loss = 1 - dice_per_class[1:].mean()

        return dice_loss

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        loss = self.weight_ce * ce + (1 - self.weight_ce) * dice
        # print all three losses
        return loss, ce.item(), dice.item()
    

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedDiceCELoss(nn.Module):
    """
    Multiclass Dice + Cross Entropy Loss
    - Soft Dice on probabilities
    - Class-weighted Dice (foreground only)
    - Background excluded from Dice
    - CE computed on all classes
    """

    def __init__(
        self,
        dice_weights: torch.Tensor,   # shape [C], includes background
        weight_ce: float = 0.5,
        smooth: float = 1.0,
        ce_weights: torch.Tensor = None,
    ):
        super().__init__()

        self.weight_ce = weight_ce
        self.smooth = smooth

        self.register_buffer("dice_weights", dice_weights)

        self.ce_loss = nn.CrossEntropyLoss(weight=ce_weights)

    def dice_loss(self, logits, targets):
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # one-hot encoding
        if logits.dim() == 4:  # 2D
            targets_oh = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
            dims = (0, 2, 3)
        elif logits.dim() == 5:  # 3D
            targets_oh = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()
            dims = (0, 2, 3, 4)
        else:
            raise ValueError(f"Unsupported shape {logits.shape}")

        intersection = torch.sum(probs * targets_oh, dims)
        denominator = torch.sum(probs + targets_oh, dims)

        dice_per_class = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        # foreground only (exclude background index 0)
        dice_fg = dice_per_class[1:]
        weights_fg = self.dice_weights[1:]

        weighted_dice = torch.sum(weights_fg * dice_fg)

        return 1.0 - weighted_dice

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        loss = self.weight_ce * ce + (1.0 - self.weight_ce) * dice
        return loss, ce.item(), dice.item()
