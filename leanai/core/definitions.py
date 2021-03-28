"""doc
# leanai.core.definitions

> Common definitions that are required for your ai development.

Datasplits
* `SPLIT_TRAIN`: Constant used to identify the training datasplit.
* `SPLIT_VAL`: Constant used to identify the validation datasplit. This happens at the end of every epoch and during development.
* `SPLIT_TEST`: Constant used to identify the test datasplit. Testing should only be done once. You should only test the best model that was identified via validation.

Phase of the process
* `PHASE_TRAIN`: Constant specifying the phase to be training of the model.
* `PHASE_VAL`: Constant specifying the phase to be validation of the model. This happens at the end of every epoch and during development.
* `PHASE_TEST`: Constant specifying the phase to be testing of the model. Testing should only be done once. You should only test the best model that was identified via validation.
"""
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"

PHASE_TRAIN = "train"
PHASE_VAL = "val"
PHASE_TEST = "test"
