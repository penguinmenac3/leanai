"""doc
# leanai.core.cli

> The command line interface for leanai.

By using this package you will not need to write your own main for most networks. This helps reduce boilerplate code.
"""
import argparse
import os
import leanai.core.logging as logging


def set_seeds():
    """
    Sets the seeds of torch, numpy and random for reproducability.
    """
    import torch
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)
    import random
    random.seed(0)


def run(experiment_class):
    """
    You can use this main function to make your experiment runnable with command line arguments.

    Simply add this to the end of your experiment.py file:

    ```python
    if __name__ == "__main__":
        from pytorch_mjolnir import run
        run(MyExperiment)
    ```

    Then you can call your python file from the command line and use the help to figure out the parameters.
    ```bash
    python my_experiment.py --help
    ```
    """
    gpus = 1
    if "SLURM_GPUS" in os.environ:
        gpus = int(os.environ["SLURM_GPUS"])
    nodes = 1
    if "SLURM_NODES" in os.environ:
        nodes = int(os.environ["SLURM_NODES"])
    output_path = "logs"
    if "RESULTS_PATH" in os.environ:
        output_path = os.environ["RESULTS_PATH"]
    
    parser = argparse.ArgumentParser(description='The main entry point for the script.')
    parser.add_argument('--name', type=str, required=True, help='The name for the experiment.')
    parser.add_argument('--version', type=str, required=False, default=None, help='The version that should be used (defaults to timestamp).')
    parser.add_argument('--output', type=str, required=False, default=output_path, help='The name for the experiment (defaults to $RESULTS_PATH or "logs").')
    parser.add_argument('--mode', type=str, required=False, default="train", help='The action that should be executed (in the config/experiment calls the function named like this).')
    parser.add_argument('--gpus', type=int, required=False, default=gpus, help='Number of GPUs that can be used.')
    parser.add_argument('--nodes', type=int, required=False, default=nodes, help='Number of nodes that can be used.')
    parser.add_argument('--resume_checkpoint', type=str, required=False, default=None, help='A specific checkpoint to load. If not provided it tries to load latest if any exists.')
    parser.add_argument('--debug', action='store_true', help='This flag will make leanai print debug messages.')
    parser.add_argument('--device', type=str, default=None, required=False, help='CUDA device id if you want to use a specific one.')
    args, other_args = parser.parse_known_args()
    kwargs = _parse_other_args(other_args)

    if args.debug:
        logging.DEBUG_VERBOSITY = args.debug
        logging.debug(f"Set DEBUG_VERBOSITY={logging.DEBUG_VERBOSITY}")
    
    if args.device:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    for k, v in args._get_kwargs():
        logging.info(f"Arg: --{k} {v}")
    for k, v in kwargs.items():
        logging.info(f"Arg: --{k} {v}")
    _instantiate_and_run(experiment_class, name=args.name, version=args.version, output=args.output, resume_checkpoint=args.resume_checkpoint, gpus=args.gpus, nodes=args.nodes, **kwargs)


def _instantiate_and_run(
        experiment_class,
        name: str,
        mode: str = "train",
        version: str = None,
        output: str = "logs",
        resume_checkpoint: str = None,
        gpus: int = 1,
        nodes: int = 1,
        **kwargs
    ):
    set_seeds()
    experiment = experiment_class(**kwargs)
    fun = getattr(experiment, f"run_{mode}", None)
    if fun is None:
        raise NotImplementedError(f"No function run_{fun} in your experiment, but --mode='{fun}' requires that function to exist.")
    else:
        return fun(name=name, version=version, output_path=output, resume_checkpoint=resume_checkpoint, gpus=gpus, nodes=nodes)


def _parse_other_args(other_args):
    kwargs = {}
    for arg in other_args:
        parts = arg.split("=")
        k = parts[0]
        if k.startswith("--"):
            k = k[2:]
        if len(parts) == 1:
            v = True
        else:
            v = "=".join(parts[1:])
            if v.startswith('"') or v.startswith("'"):
                v = v[1:-2]
            elif v == "True":
                v = True
            elif v == "False":
                v = False
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
        kwargs[k] = v
    return kwargs
