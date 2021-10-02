[Back to Overview](../../../README.md)



# leanai.data.datasets.webdataset

> Some helpers to work with webdatasets. You can use the regular wds library if you do not need these helpers.


---
### *def* **WebDataset**(tars: List[str], parser: IParser, **kwargs) -> IIterableDataset

Create a webdataset using tars and a paraser.

Advanced users might want to use webdataset directly.

* **tars**: A list of tar files that should be loaded.
* **parser**: A parsing function that can be used to convert bytes to samples.
* ****kwargs**: Any arguments that webdataset.WebDataset accepts, a common one would be `shardshuffle=True`.


---
### *def* **create_archive**(output_filename: str, filenames: Iterable[str], mapping: Callable[[str], str] = _*def*ault_mapping, zip_compression=False)

Create an archive given filenames.

* **output_filename**: The output filename of the archive (.tar, .tar.gz and .zip are supported).
* **filenames**: A stream of filenames that should be put into the archive in order, make sure this is the order in which you want to read the data.
Note that shuffling on read is limited to an in memory buffer. Thus, large datasets have limited randomness.
* **mapping**: A function that maps real filenames to the wds conform name "{sample_token}.{annotype}.{extension}" (defaults to identity).
* **zip_compression**: A boolean if the zip should be compressed. (Use with caution as it is slow).


