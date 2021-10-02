"""doc
# leanai.data.datasets.webdataset

> Some helpers to work with webdatasets. You can use the regular wds library if you do not need these helpers.
"""
from typing import Iterable, Callable, Dict, List
import os
import zipfile
import tarfile
import webdataset as wds

from leanai.data.dataset import IIterableDataset
from leanai.data.parser import IParser
from leanai.data.data_promise import DataPromise, DataPromiseFromBytes


def _promise_wrapping(sample: Dict[str, any]) -> Dict[str, DataPromise]:
    return {
        k: DataPromiseFromBytes(v)
        for k, v in sample.items()
    }


def _default_mapping(fname: str) -> str:
    # Mapps the real filename to a wds name.
    # :param fname: The real filename.
    # :return: The filename in the wds archive.
    return fname


def WebDataset(tars: List[str], parser: IParser, **kwargs) -> IIterableDataset:
    """
    Create a webdataset using tars and a paraser.

    Advanced users might want to use webdataset directly.

    :param tars: A list of tar files that should be loaded.
    :param parser: A parsing function that can be used to convert bytes to samples.
    :param **kwargs: Any arguments that webdataset.WebDataset accepts, a common one would be `shardshuffle=True`.
    """
    dataset = wds.WebDataset(tars, **kwargs)
    dataset = wds.Processor(dataset, wds.map, _promise_wrapping)
    dataset = wds.Processor(dataset, wds.map, parser)
    return dataset


def create_archive(output_filename: str, filenames: Iterable[str], mapping: Callable[[str], str] = _default_mapping, zip_compression=False):
    """
    Create an archive given filenames.

    :param output_filename: The output filename of the archive (.tar, .tar.gz and .zip are supported).
    :param filenames: A stream of filenames that should be put into the archive in order, make sure this is the order in which you want to read the data.
        Note that shuffling on read is limited to an in memory buffer. Thus, large datasets have limited randomness.
    :param mapping: A function that maps real filenames to the wds conform name "{sample_token}.{annotype}.{extension}" (defaults to identity).
    :param zip_compression: A boolean if the zip should be compressed. (Use with caution as it is slow).
    """
    base_path = os.path.dirname(output_filename)
    os.makedirs(base_path, exist_ok=True)
    if output_filename.endswith(".zip"):
        with zipfile.ZipFile(output_filename,"w") as archive:
            if zip_compression:
                compression = zipfile.ZIP_DEFLATED
            else:
                compression = zipfile.ZIP_STORED
            for fname in filenames:
                archive.write(fname, arcname=mapping(fname), compress_type=compression)
    elif output_filename.endswith(".tar.gz"):
        with tarfile.open(output_filename,"w:gz") as archive:
            for fname in filenames:
                archive.add(fname, arcname=mapping(fname))
    elif output_filename.endswith(".tar"):
        with tarfile.open(output_filename,"w") as archive:
            for fname in filenames:
                archive.add(fname, arcname=mapping(fname))
    else:
        raise RuntimeError("Filename must end with '.zip' or '.tar.gz'! Found: {}".format(output_filename))
