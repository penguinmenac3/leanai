from deeptech.training.losses.classification import BinaryCrossEntropyLossFromLogits, SparseCrossEntropyLossFromLogits, SparseCategoricalAccuracy
from deeptech.training.losses.masking import NaNMaskedLoss, NegMaskedLoss, MaskedLoss
from deeptech.training.losses.regression import SmoothL1Loss
from deeptech.training.losses.multiloss import MultiLoss
from deeptech.training.losses.detection import DetectionLoss
