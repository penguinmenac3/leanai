"""doc
# deeptech.core.checkpoint

> Loading and saving checkpoints with babilim.
"""
import os
import numpy as np
from typing import Dict
from deeptech.core import logging


def init_model(model, dataloader):
    """
    Initialize the model, so that the checkpoints can be loaded.

    This is done by running the forward pass once with an example.

    :param model: The model that should be initialized.
    :param dataloader: A dataloader that can be used to get a sample to feed through the network for full initialization.
    """
    if not getattr(model, "initialized_model", False):
        if logging.DEBUG_VERBOSITY:
            logging.info("Build Model")
        model.initialized_model = True
        features, _ = next(iter(dataloader))
        model(*features)


def load_state(checkpoint_path: str) -> Dict:
    """
    Load the state from a checkpoint.
    
    :param checkpoint_path: The path to the file in which the checkpoint is stored.
    :return: A dict containing the states.
    """
    if not os.path.exists(checkpoint_path):
        raise RuntimeError("Checkpoint path does not exist. Please provide a path to a *.pth or *.npy file. You provided: {}".format(checkpoint_path))
    if checkpoint_path.endswith(".pth"):
        import torch
        return torch.load(checkpoint_path, map_location='cpu')
    elif checkpoint_path.endswith(".npz"):
        data = np.load(checkpoint_path, allow_pickle=True)
        out = {}
        prefixes = list(set([key.split("/")[0] for key in list(data.keys())]))
        for prefix in prefixes:
            if prefix in data:  # primitive types
                out[prefix] = data[prefix]
            else:  # dict types
                tmp = {"{}".format("/".join(k.split("/")[1:])): data[k] for k in data if k.startswith(prefix)}
                out[prefix] = tmp
        return out
    else:
        raise NotImplementedError()


def load_weights(checkpoint_path: str, model):
    """
    Load the weights from a checkpoint into a model.
    
    :param checkpoint_path: The path to the file in which the checkpoint is stored.
    :param model: The model for which to set the state from the checkpoint.
    """
    checkpoint = load_state(checkpoint_path)
    if "model" in checkpoint:
        if logging.DEBUG_VERBOSITY:
            logging.info("Load Model...")
        model.load_state_dict(checkpoint["model"])
    else:
        logging.warn("Could not find model_state in checkpoint.")


def save_state(data, checkpoint_path, file_format: str = "numpy"):
    """
    Save the state to a checkpoint.
    
    :param data: A dict containing the states.
    :param checkpoint_path: The path to the file in which the checkpoint shall be stored.
    :param file_format: (Optional[str]) The format in which the checkpoint should be stored. (Default: "numpy")
    """
    if file_format == "pytorch":
        import torch
        return torch.save(data, f"{checkpoint_path}.pth")
    elif file_format == "numpy":
        out = {}
        for key, value in data.items():
            if isinstance(value, dict):
                tmp = {"{}/{}".format(key, k): value[k] for k in value}
                out.update(tmp)
            elif any(isinstance(value, t) for t in [int, str, float, list]):
                out[key] = value
            else:
                raise RuntimeError("The type ({}) of {} is not allowed!".format(type(value), key))
        np.savez_compressed(f"{checkpoint_path}.npz", **out)
    else:
        raise NotImplementedError()
