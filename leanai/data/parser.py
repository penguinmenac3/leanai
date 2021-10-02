"""doc
# leanai.data.parser

> The interface and a reference implementation for parsing samples into a usable format.
"""
from typing import Any, Dict, Tuple
import io
import json
import numpy as np
from .data_promise import DataPromise


class IParser(object):
    """
    Interface for a parser.

    A parser must implement the `__call__` method.
    ```python
    def __call__(self, sample: Dict[str, DataPromise]) -> Any:
    ````
    The call method gets the file promises loaded by the file provider and parses them into usable formats.
    """
    def __call__(self, sample: Dict[str, DataPromise]) -> Any:
        raise NotImplementedError("Must be implemented by subclass!")


class Parser(IParser):
    def __init__(self, InputType, OutputType, ignore_file_not_found=False) -> None:
        """
        Parser decodes the samples required to populate the input and output type automatically.

        The Parser should never be instantiated directly.
        You should inherit it and then instantiate the inherited class.

        The Parser automatically only parses those examples which are used in the InputType and OutputType.
        Data which is not used in any of the two will not be parsed thus speeding up the parsing process.

        To achieve this, the parser tries to call functions called "parse_{attribute_name}" (and in case that fails "get_{attribute_name}"),
        where "attribute_name" is the field name in the named tuple.

        For example providing this named tuple:
        `InputType = NamedTuple("InputType", image=np.ndarray)`
        will result in a call to `parse_image` or an error if that function does not exist.
        It will be assumed, that `parse_image` returns an `np.ndarray`.
        If that is not the case, your code may break somewhere.

        **Arguments**
        :param InputType: A definition of a named tuple that defines the input of the neural network.
        :param OutputType: A definition of a named tuple that defines the output of the neural network.
        :param ignore_file_not_found: If a file is missing return None instead of an exception.  (Default: False).
        """
        self.dataset_input_type = InputType
        self.dataset_output_type = OutputType
        self.ignore_file_not_found = ignore_file_not_found

    def __call__(self, sample: Dict[str, DataPromise]) -> Tuple[Any, Any]:
        dataset_input = self._fill_type_using_parsers(self.dataset_input_type, sample)
        dataset_output = self._fill_type_using_parsers(self.dataset_output_type, sample)
        return dataset_input, dataset_output

    def _fill_type_using_parsers(self, namedtuple_type, sample):
        data = {}
        for k in namedtuple_type._fields:
            parser = getattr(self, "parse_{}".format(k), None)
            if parser is None:
                parser = getattr(self, "get_{}".format(k), None)
            if parser is not None:
                try:
                    data[k] = parser(sample)
                except FileNotFoundError as e:
                    if self.ignore_file_not_found:
                        data[k] = None
                    else:
                        raise e
            else:
                raise RuntimeError("Missing parser (parse_{}) for dataset_input_type field: {}".format(k, k))
        return namedtuple_type(**data)

    @staticmethod
    def decode_image(data: bytes, mode: str = "RGB") -> np.ndarray:
        """
        Static method, that allows for decoding an image from bytes in the byte dict.

        This function will be used in parsers, e.g.:
        ```python
        def parse_image(self, sample) -> np.ndarray:
            return BaseParser.decode_image(sample["image"])
        ```

        :param data: Bytes that represent an image. The bytes can be received from a sample in a parse_X function.
        :param mode: The mode in which the image should be laoded. (Default: "RGB")
        :return: An image as a np.ndarray.
        """
        import PIL.Image
        with io.BytesIO(data) as stream:
            img = PIL.Image.open(stream)
            img.load()
            img = img.convert(mode.upper())
        return np.asarray(img)

    @staticmethod
    def decode_numpy(data: bytes, dtype=np.float32) -> np.ndarray:
        """
        Static method, that allows for decoding a numpy file from bytes in the byte dict.

        This function will be used in parsers, e.g.:
        ```python
        def parse_scan(self, sample) -> np.ndarray:
            return BaseParser.decode_numpy(sample["scan"])
        ```

        :param data: Bytes that represent an npy file. The bytes can be received from a sample in a parse_X function.
        :param dtype: The dtype the data has. (Default: np.float32)
        :return: A numpy array.
        """
        return np.frombuffer(data, dtype=dtype)

    @staticmethod
    def decode_text(data: bytes, encoding="utf-8") -> str:
        """
        Static method, that allows for decoding a txt file from bytes in the byte dict.

        This function will be used in parsers, e.g.:
        ```python
        def parse_text(self, sample) -> str:
            return BaseParser.decode_text(sample["label"])
        ```

        :param data: Bytes that represent a txt file. The bytes can be received from a sample in a parse_X function.
        :param encoding: The encoding of the text. (Default: "utf-8")
        :return: The text of the file.
        """
        return data.decode(encoding)

    @staticmethod
    def decode_json(data: bytes, encoding="utf-8"):
        """
        Static method, that allows for decoding a json file from bytes in the byte dict.

        This function will be used in parsers, e.g.:
        ```python
        def parse_annotation(self, sample):
            return BaseParser.decode_json(sample["annotation"])
        ```

        :param data: Bytes that represent a json file. The bytes can be received from a sample in a parse_X function.
        :param encoding: The encoding of the text. (Default: "utf-8")
        :return: A (dict/list) object representing the content of the json.
        """
        return json.loads(Parser.decode_text(data, encoding))
