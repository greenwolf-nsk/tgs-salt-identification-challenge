import torch
import torch.nn as nn

from .lovasz_loss import lovasz_hinge


class BoundaryLoss:

    def __init__(self, weights: tuple = (0.1, 1)):
        self.boundary_loss_fn = nn.BCEWithLogitsLoss()
        self.segmentation_loss_fn = nn.BCEWithLogitsLoss()
        self.weights = weights

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor):
        segmentation_loss = self.boundary_loss_fn(outputs[:, :1, :, :], targets[:, :1, :, :])
        boundary_loss = self.boundary_loss_fn(outputs[:, 1:, :, :], targets[:, 1:, :, :])

        return (
            self.weights[0] * segmentation_loss +
            self.weights[1] * boundary_loss
        )


class BoundaryLoss2:

    def __init__(self, weights: tuple = (0.1, 1)):
        self.boundary_loss_fn = nn.BCEWithLogitsLoss()
        self.segmentation_loss_fn = lovasz_hinge
        self.weights = weights

    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor):
        segmentation_loss = self.boundary_loss_fn(outputs[:, :1, :, :], targets[:, :1, :, :])
        boundary_loss = self.boundary_loss_fn(outputs[:, 1:, :, :], targets[:, 1:, :, :])

        return (
            self.weights[0] * segmentation_loss +
            self.weights[1] * boundary_loss
        )


class CombinedLoss:

    def __init__(self, segmentation_loss_fn, weights: tuple = (0.05, 0.1, 1.0)):
        self.classification_loss_fn = nn.BCEWithLogitsLoss()
        self.non_empty_segmentation_loss_fn = segmentation_loss_fn
        self.segmentation_loss_fn = segmentation_loss_fn
        self.weights = weights

    def __call__(self, outputs: tuple, targets: torch.Tensor):
        fused_outputs, clf_outputs, seg_outputs = outputs
        classes = targets.sum(dim=(1, 2, 3)) > 0
        classification_loss = self.classification_loss_fn(clf_outputs, classes.float().unsqueeze(1))
        non_empty_segmentation_loss = self.segmentation_loss_fn(seg_outputs[classes], targets[classes])
        segmentation_loss = self.segmentation_loss_fn(fused_outputs, targets)
        return (
            self.weights[0] * classification_loss +
            self.weights[1] * non_empty_segmentation_loss +
            self.weights[2] * segmentation_loss
        )


class CombinedLossHypercolumn:

    def __init__(self, segmentation_loss_fn, weights: tuple = (0.05, 0.1, 1.0)):
        self.classification_loss_fn = nn.BCEWithLogitsLoss()
        self.non_empty_segmentation_loss_fn = segmentation_loss_fn
        self.segmentation_loss_fn = segmentation_loss_fn
        self.weights = weights

    def __call__(self, outputs: tuple, targets: torch.Tensor):
        fused_outputs, clf_outputs, seg_outputs = outputs
        classes = targets.sum(dim=(1, 2, 3)) > 0
        classification_loss = self.classification_loss_fn(clf_outputs, classes.float().unsqueeze(1))
        non_empty_segmentation_loss = 0
        for seg_output in seg_outputs:
            non_empty_segmentation_loss += self.segmentation_loss_fn(seg_output[classes], targets[classes])
        non_empty_segmentation_loss /= len(seg_outputs)

        segmentation_loss = self.segmentation_loss_fn(fused_outputs, targets)
        return (
            self.weights[0] * classification_loss +
            self.weights[1] * non_empty_segmentation_loss +
            self.weights[2] * segmentation_loss
        )


