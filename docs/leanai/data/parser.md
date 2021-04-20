[Back to Overview](../../README.md)



# leanai.data.parser

> The interface and a reference implementation for parsing samples into a usable format.


---
---
## *class* **IParser:*

Interface for a parser.

A parser must implement the __call__ method.
```python
def __call__(self, sample: Dict[str, DataPromise]) -> Any:
````
The call method gets the file promises loaded by the file provider and parses them into usable formats.


---
---
## *class* **Parser**(I**Parser**)

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
* **InputType**: A definition of a named tuple that defines the input of the neural network.
* **OutputType**: A definition of a named tuple that defines the output of the neural network.


---
### *def* **decode_image**(data: bytes, mode: str = "RGB") -> np.ndarray

Static method, that allows for decoding an image from bytes in the byte dict.

This function will be used in parsers, e.g.:
```python
def parse_image(self, sample) -> np.ndarray:
return BaseParser.decode_image(sample["image"])
```

* **data**: Bytes that represent an image. The bytes can be received from a sample in a parse_X function.
* **mode**: The mode in which the image should be laoded. (Default: "RGB")
* **returns**: An image as a np.ndarray.


---
### *def* **decode_numpy**(data: bytes, dtype=np.float32) -> np.ndarray

Static method, that allows for decoding a numpy file from bytes in the byte dict.

This function will be used in parsers, e.g.:
```python
def parse_scan(self, sample) -> np.ndarray:
return BaseParser.decode_numpy(sample["scan"])
```

* **data**: Bytes that represent an npy file. The bytes can be received from a sample in a parse_X function.
* **dtype**: The dtype the data has. (Default: np.float32)
* **returns**: A numpy array.


---
### *def* **decode_text**(data: bytes, encoding="utf-8") -> str

Static method, that allows for decoding a txt file from bytes in the byte dict.

This function will be used in parsers, e.g.:
```python
def parse_text(self, sample) -> str:
return BaseParser.decode_text(sample["label"])
```

* **data**: Bytes that represent a txt file. The bytes can be received from a sample in a parse_X function.
* **encoding**: The encoding of the text. (Default: "utf-8")
* **returns**: The text of the file.


---
### *def* **decode_json**(data: bytes, encoding="utf-8")

Static method, that allows for decoding a json file from bytes in the byte dict.

This function will be used in parsers, e.g.:
```python
def parse_annotation(self, sample):
return BaseParser.decode_json(sample["annotation"])
```

* **data**: Bytes that represent a json file. The bytes can be received from a sample in a parse_X function.
* **encoding**: The encoding of the text. (Default: "utf-8")
* **returns**: A (dict/list) object representing the content of the json.


