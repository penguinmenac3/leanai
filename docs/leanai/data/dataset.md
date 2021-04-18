[Back to Overview](../../README.md)



# leanai.data.dataset

> A generic implementation for a dataset based on parsers and file providers.


---
---
## *class* **IIterableDataset**(_IterableDataset)

Interface for an iterable dataset.

Has iter and next.


---
---
## *class* **ISequenceDataset**(_Dataset)

Interface for a sequence dataset.

Has len, getitem, iter and next.


---
---
## *class* **IterableDataset**(_CommonDataset, I**IterableDataset**)

An implementation of the IIterableDataset using fileprovider and parser.

* **file_provider_iterable**: The iterable file provider providing samples to the parser.
* **parser**: The parser converting samples into a usable format.


---
---
## *class* **SequenceDataset**(_CommonDataset, I**SequenceDataset**)

An implementation of the ISequenceDataset using fileprovider and parser.

* **file_provider_sequence**: The sequence file provider providing samples to the parser.
* **parser**: The parser converting samples into a usable format.


---
---
## *class* **SimpleDataset**(Parser, SequenceDataset)

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
* **InputType**: A definition of a named tuple that defines the input of the neural network.
* **OutputType**: A definition of a named tuple that defines the output of the neural network.


---
### *def* **set_sample_tokens**(*self*, sample_tokens: List[Any]) -> None

Set the list of sample tokens.

* **sample_tokens**: A list of all sample tokens of the dataset.


