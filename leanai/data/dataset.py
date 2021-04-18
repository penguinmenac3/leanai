"""doc
# leanai.data.dataset

> A generic implementation for a dataset based on parsers and file providers.
"""
from typing import Any, Dict, Iterator, List
from torch.utils.data import IterableDataset as _IterableDataset
from torch.utils.data import Dataset as _Dataset

from .parser import IParser, Parser
from .file_provider import FileProviderSequence, FileProviderIterable
from .data_promise import DataPromise


class IIterableDataset(_IterableDataset):
    """
    Interface for an iterable dataset.

    Has iter and next.
    """
    def __next__(self) -> Any:
        raise NotImplementedError("Must be implemented by subclass.")

    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError("Must be implemented by subclass.")


class ISequenceDataset(_Dataset):
    """
    Interface for a sequence dataset.

    Has len, getitem, iter and next.
    """
    def __next__(self) -> Any:
        raise NotImplementedError("Must be implemented by subclass.")

    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError("Must be implemented by subclass.")

    def __getitem__(self, index) -> Any:
        raise NotImplementedError("Must be implemented by subclass.")

    def __len__(self) -> int:
        raise NotImplementedError("Must be implemented by subclass.")


class _CommonDataset:
    def __init__(self, file_provider_iterable: FileProviderIterable, parser: IParser) -> None:
        super().__init__()
        self._file_provider = file_provider_iterable
        self._fp_iterator = None
        self._parser = parser
        self.transformers = []

    def _process(self, sample: Dict[str, DataPromise]) -> Any:
        sample = self._parser(sample)
        for transformer in self.transformers:
            sample = transformer(sample)
        return sample
    
    def __next__(self) -> Any:
        if self._fp_iterator is None:
            raise RuntimeError("You must first call iter(...) before you can use next(...).")
        sample = self._fp_iterator.__next__()
        return self._process(sample)

    def __iter__(self) -> Iterator[Any]:
        self._fp_iterator = self._file_provider.__iter__()
        return self

    def __len__(self) -> int:
        return len(self._file_provider)


class IterableDataset(_CommonDataset, IIterableDataset):
    def __init__(self, file_provider_iterable: FileProviderIterable, parser: IParser) -> None:
        """
        An implementation of the IIterableDataset using fileprovider and parser.

        :param file_provider_iterable: The iterable file provider providing samples to the parser.
        :param parser: The parser converting samples into a usable format.
        """
        super().__init__(file_provider_iterable, parser)


class SequenceDataset(_CommonDataset, ISequenceDataset):
    def __init__(self, file_provider_sequence: FileProviderSequence, parser: IParser) -> None:
        """
        An implementation of the ISequenceDataset using fileprovider and parser.

        :param file_provider_sequence: The sequence file provider providing samples to the parser.
        :param parser: The parser converting samples into a usable format.
        """
        super().__init__(file_provider_sequence, parser)
        self._file_provider = file_provider_sequence

    def __getitem__(self, index) -> Any:
        sample = self._file_provider[index]
        return self._process(sample)

class SimpleDataset(Parser, SequenceDataset):
    def __init__(self, InputType, OutputType) -> None:
        """
        SimpleDataset decodes the samples required to populate the input and output type automatically.

        The SimpleDataset should never be instantiated directly.
        You should inherit it and then instantiate the inherited class.

        The SimpleDataset automatically only parses those examples which are used in the InputType and OutputType.
        Data which is not used in any of the two will not be parsed thus speeding up the parsing process.

        To achieve this, the parser tries to call functions called "parse_{attribute_name}",
        where "attribute_name" is the field name in the named tuple.

        For example providing this named tuple:
        `InputType = NamedTuple("InputType", image=np.ndarray)`
        will result in a call to `parse_image` or an error if that function does not exist.
        The following signature will be expected:
        ```
        def parse_image(self, sample_token) -> np.ndarray:
        ```
        If that is not the case, your code may break somewhere.

        **Arguments**
        :param InputType: A definition of a named tuple that defines the input of the neural network.
        :param OutputType: A definition of a named tuple that defines the output of the neural network.
        """
        Parser.__init__(self, InputType, OutputType)
        SequenceDataset.__init__(self, [], self)

    def set_sample_tokens(self, sample_tokens: List[Any]) -> None:
        """
        Set the list of sample tokens.

        :param sample_tokens: A list of all sample tokens of the dataset.
        """
        self._file_provider = list(sample_tokens)
