"""doc
# leanai.core.logging

> This is a logger taking care of logging the code, console outputs and images. It does not log tensors in tensorboard though. If you want to log to tensorboard use `experiment.log` instead of logging functions here.

## Outline

This package helps with logging while training and managing checkpoints (creation, eta estimation, etc.).

1. Global Flags
2. Time Helpers
3. Logging
"""
import os
import sys
import shutil
import time
import fnmatch
from typing import List, Union
from functools import reduce
from pytorch_lightning.utilities.rank_zero import rank_zero_only

import time as __time
import datetime as __datetime


"""doc
# Global Flags

These flags allow controlling what gets printed when using status, info, warn and error function. Debug verbosity enables a lot of logging leanai internally.

```python
DEBUG_VERBOSITY = False
PRINT_STATUS = True
PRINT_INFO = True
PRINT_WARN = True
PRINT_ERROR = True
```

For logging the code an ignore list is used to find out what files to ignore. If you need to change those files you can modify the following global variable of this module, before setting up the logger.

```python
PYTHON_IGNORE_LIST = ["__pycache__", "*.pyc", ".ipynb_checkpoints", "checkpoints", "logs", "dist", "docs", "*.egg-info", "tfrecords", "*.code-workspace", ".git"]
```

For example:

```python
from leanai.core import logging
logging.PYTHON_IGNORE_LIST.append("*.my_extension")
logging.set_logger("...")
```
"""
__logfile = None
__log_buffer = []


DEBUG_VERBOSITY = 0
DEBUG_LEVEL_EXTERNAL = 1
DEBUG_LEVEL_API = 10
DEBUG_LEVEL_CORE = 20

FILE_LOG_LEVEL = 0
PRINT_STATUS = True
PRINT_INFO = True
PRINT_WARN = True
PRINT_ERROR = True

PYTHON_IGNORE_LIST = ["__pycache__", "*.pyc", ".ipynb_checkpoints", "checkpoints", "logs", "dist", "docs", "*.egg-info",
                      "tfrecords", "*.code-workspace", ".git"]

"""doc
# Time Helpers

These functions help handling time formating and computations.
"""
def format_time(t: float) -> str:
    """
    Format a time duration in a readable format.
    
    :param t: The duration in seconds.
    :return: A human readable string.
    """
    hours, remainder = divmod(t, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%d:%02d:%02d' % (hours, minutes, seconds)


def get_timestamp() -> str:
    """
    Create a string for the current timestamp.
    
    :return: Current date and time as a string suitable for a logfolder filename.
    """
    time_stamp = __datetime.datetime.fromtimestamp(__time.time()).strftime('%Y-%m-%d_%H.%M.%S')
    return time_stamp


"""doc
# Logging

These functions help with actually writing stuff to the logfolder and setting it up.
"""
def __get_all_files(root: str = None, forbidden_list: List[str] = PYTHON_IGNORE_LIST) -> List[str]:
    if root is None:
        root = os.getcwd()
    
    def _is_not_in_forbidden_list(candidate: str) -> bool:
        candidate = candidate.replace("\\", "/")
        res = map(lambda x: not fnmatch.fnmatch(candidate, x), forbidden_list)
        return reduce(lambda x, y: (x and y), res)

    def _filter_files(files):
        return list(filter(_is_not_in_forbidden_list, files))
    
    def _filter_subdirs(subdirs):
        filtered = filter(_is_not_in_forbidden_list, subdirs)
        return list(filter(lambda x: not x.startswith("."), filtered))
    
    def _join_filenames(root, path, files):
        root_with_sep = root + os.sep
        files = map(lambda name: os.path.join(path, name).replace(root_with_sep, ""), files)
        return list(files)
    
    all_files = []
    for path, subdirs, files in os.walk(root):
        files = _filter_files(files)
        subdirs[:] = _filter_subdirs(subdirs)
        all_files.extend(_join_filenames(root, path, files))
    return all_files


@rank_zero_only
def _log_code(output_dir: str, overwrite_existing=False) -> None:
    """
    Log the code of the current working directory into output directory.

    :param output_dir: The directory where to copy all code.
    :param overwrite_existing: When set to true it overwrites existing code copies.
    """
    base_path = os.path.join(os.getcwd())
    def _get_backup_path(fname: str) -> str:
        return os.path.join(os.path.normpath(output_dir), fname)

    def _create_missing_dir(fname: str) -> None:
        path = os.path.dirname(fname)
        if not os.path.exists(path):
            os.makedirs(path)

    def _backup(fname: str, base_path: str) -> None:
        target = _get_backup_path(fname)
        _create_missing_dir(target)
        shutil.copyfile(os.path.join(base_path, fname), target)

    if overwrite_existing or not os.path.exists(output_dir):
        for f in __get_all_files(base_path):
            _backup(f, base_path)


def set_logger(log_folder: str, log_code: bool = True) -> None:
    """
    Setup the logger.

    Creates the log folder, a src folder inside the log folder where it copies the current working directory.

    :param log_folder: Folder is is used for storing all logs (code, checkpoints, images, text).
    """
    def _set_logfile(log_file):
        global __logfile
        if __logfile is not None:
            raise RuntimeError("You must not setup logger twice!")
        __logfile = log_file

    def _create_log_folder():
        log_folder = get_log_path()
        assert log_folder is not None
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
    
    def _flush_log_buffer():
        assert __logfile is not None
        with open(__logfile, "a") as f:
            for data in __log_buffer:
                f.write(data + "\n")

    debug(f"Initializing logger (log_code={log_code})", level=DEBUG_LEVEL_CORE)
    _set_logfile(os.path.join(log_folder, "log.txt"))
    _create_log_folder()
    _flush_log_buffer()
    log_folder = get_log_path()
    assert log_folder is not None
    if log_code:
        _log_code(os.path.join(log_folder, "src_{}".format(get_timestamp())))
    debug("Logger initialized", level=DEBUG_LEVEL_CORE)


def close() -> None:
    """
    Close the logger for a given reason.
    
    If none is provided there is no final progress written. Provide a reason, if you do not manually set the final progress before.
    A training loop typically manually sets the progress, so you will not need a reason in that case.
    
    :param reason: The reason for the closing of the logger. It is recommended to use "done", "paused", "failed" as reason.
    """
    global __logfile
    global __log_buffer
    if __logfile is None:
        raise RuntimeError("You must setup the logger before you can close it!")

    __log_buffer = []
    __logfile = None


def get_log_path() -> Union[str, None]:
    """
    Gets the log path based on the logfile.
    
    :return: The path containing the logfile.
    """
    if __logfile is None:
        return None
    return os.path.dirname(__logfile)


def create_checkpoint_structure() -> None:
    """
    Create a checkpoint structure in the log folder.
    
    * train: Folder for train split tensorboard logs.
    * val: Folder for val split tensorboard logs.
    * test: Folder for test split tensorboard logs.
    * checkpoints: Folder for the leanai, pytorch or tensorboard checkpoints.
    """
    logfolder = get_log_path()
    assert logfolder is not None
    if not os.path.exists(os.path.join(logfolder, "train")):
        os.makedirs(os.path.join(logfolder, "train"))
    if not os.path.exists(os.path.join(logfolder, "val")):
        os.makedirs(os.path.join(logfolder, "val"))
    if not os.path.exists(os.path.join(logfolder, "test")):
        os.makedirs(os.path.join(logfolder, "test"))
    if not os.path.exists(os.path.join(logfolder, "checkpoints")):
        os.makedirs(os.path.join(logfolder, "checkpoints"))


def status(msg: str, end: str = "\n") -> None:
    """
    Print something with a timestamp.
    Useful for logging.
    Leanai internally uses this for all its log messages.

    :param msg: The message to print.
    :param end: The line ending. Defaults to "\n" but can be set to "" to not have a linebreak.
    """
    if PRINT_STATUS:
        time_stamp = __datetime.datetime.fromtimestamp(__time.time()).strftime('%Y-%m-%d %H:%M:%S')
        data = "[{}] STAT {}".format(time_stamp, msg)
        print("\r{}".format(data), end=end)
        sys.stdout.flush()
        if end != "" and FILE_LOG_LEVEL >= 0:
            if __logfile is not None:
                with open(__logfile, "a") as f:
                    f.write(data + "\n")
            else:
                __log_buffer.append(data)


def info(msg: str, end: str= "\n") -> None:
    """
    Print something with a timestamp.
    Useful for logging.
    Leanai internally uses this for all its log messages.

    :param msg: The message to print.
    :param end: The line ending. Defaults to "\n" but can be set to "" to not have a linebreak.
    """
    if PRINT_INFO:
        time_stamp = __datetime.datetime.fromtimestamp(__time.time()).strftime('%Y-%m-%d %H:%M:%S')
        data = "[{}] INFO {}".format(time_stamp, msg)
        print("\r{}".format(data), end=end)
        sys.stdout.flush()
        if end != "" and FILE_LOG_LEVEL >= 0:
            if __logfile is not None:
                with open(__logfile, "a") as f:
                    f.write(data + "\n")
            else:
                __log_buffer.append(data)


def warn(msg: str, end: str = "\n") -> None:
    """
    Print something with a timestamp.
    Useful for logging.
    Leanai internally uses this for all its log messages.

    :param msg: The message to print.
    :param end: The line ending. Defaults to "\n" but can be set to "" to not have a linebreak.
    """
    if PRINT_WARN:
        time_stamp = __datetime.datetime.fromtimestamp(__time.time()).strftime('%Y-%m-%d %H:%M:%S')
        data = "[{}] WARN {}".format(time_stamp, msg)
        print("\r{}".format(data), end=end)
        sys.stdout.flush()
        if end != "" and FILE_LOG_LEVEL >= 0:
            if __logfile is not None:
                with open(__logfile, "a") as f:
                    f.write(data + "\n")
            else:
                __log_buffer.append(data)


def debug(msg: str, end: str = "\n", level=DEBUG_LEVEL_EXTERNAL) -> None:
    """
    Print something with a timestamp.
    Useful for logging.
    Leanai internally uses this for all its log messages.

    :param msg: The message to print.
    :param end: The line ending. Defaults to "\n" but can be set to "" to not have a linebreak.
    """
    if DEBUG_VERBOSITY >= level:
        time_stamp = __datetime.datetime.fromtimestamp(__time.time()).strftime('%Y-%m-%d %H:%M:%S')
        data = "[{}] DEBUG {}".format(time_stamp, msg)
        print("\r{}".format(data), end=end)
        sys.stdout.flush()
        if end != "" and FILE_LOG_LEVEL >= level:
            if __logfile is not None:
                with open(__logfile, "a") as f:
                    f.write(data + "\n")
            else:
                __log_buffer.append(data)


def error(msg: str, end: str = "\n") -> None:
    """
    Print something with a timestamp.
    Useful for logging.
    Leanai internally uses this for all its log messages.

    :param msg: The message to print.
    :param end: The line ending. Defaults to "\n" but can be set to "" to not have a linebreak.
    """
    if PRINT_ERROR:
        time_stamp = __datetime.datetime.fromtimestamp(__time.time()).strftime('%Y-%m-%d %H:%M:%S')
        data = "[{}] ERROR {}".format(time_stamp, msg)
        print("\r{}".format(data), end=end)
        sys.stdout.flush()
        if end != "" and FILE_LOG_LEVEL >= 0:
            if __logfile is not None:
                with open(__logfile, "a") as f:
                    f.write(data + "\n")
            else:
                __log_buffer.append(data)
