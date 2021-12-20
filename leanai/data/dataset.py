"""doc
# leanai.data.dataset

> A generic implementation for a dataset based on parsers and file providers.

Leanai Datasets generally work by you providing an input and an output type and the implementation of the
dataset filles these fields using getters and parsers.

A simple example for 2d object detection using a SimpleDataset would look like this.
```python
# Define Types
InputType = namedtuple("InputType", ["image"])
OutputType = namedtuple("OutputType", ["class_ids", "boxes_2d"])

# Instantiate Dataset
dataset = MySimpleDataset(..., InputType, OutputType, ...)

# Get Items (of types defined above)
for inp: InputType, outp: OutputType in dataset:
```

A simple example for 2d object detection using a WebDataset would look like this.
```python
# Define Types
InputType = namedtuple("InputType", ["image"])
OutputType = namedtuple("OutputType", ["class_ids", "boxes_2d"])

# Instantiate Dataset
file_provider = WebDatasetFileProvider(...)
parser = MyParser(InputType, OutputType)
dataset = IterableDataset(file_provider, parser)

# Get Items (of types defined above)
for inp: InputType, outp: OutputType in dataset:
```

This makes using datasets very easy and for implementation you only need to implement getters for the
fields which you want to access. `get_image(self, sample)` will implement the support for the `image`
field. In order to make switching between datasets easier and make behaviour predicatble, there is a
set of conventions to which a dataset implementation should adhere. The fields and their content should
adhere to the specification in the tables below.


## Object Based Data (per object, i.e. per car, per pedestrian, ...)

N represents the number of objects and the indices of the arrays allign.
If an invalid value is required for padding, NaN for Float and -1 for uint shall be used

| Name          | Shape | Description                                                                    |
|---------------|-------|--------------------------------------------------------------------------------|
| confidence    | N,1   | Float between 0 and 1. (0 = no confidence, 1 = sure)                           |
| fg_bg_classes | N,1   | uint8 (0,1) where 1 means foreground, 0 means background                       |
| class_ids     | N,1   | uint8 representing the class (0 = Background, 1 = Class 1, 2 = Class 2, ...)   |
| instance_ids  | N,1   | uint8/uint16 for InstanceID                                                    |
| occlusion     | N,1   | Float representing occlusion rate. (0 = perfectly visible, 1 = fully occluded) |
| cosy          | N,str | Name of the coordinate system in which the data is                             |
| boxes_2d      | N,4/5 | Centerpoint Representation (c_x, c_y, w, h, theta)                             |
| boxes_3d      | N,7   | Centerpoint Representation (c_x, c_y, c_z, l, w, h, theta)                     |
| boxes_3d      | N,10  | Centerpoint + Quaternion (c_x, c_y, c_z, l, w, h, w, q0, q1, q2)               |
| velocity      | N,2/3 | Velocity of the object in meters per second (c_x, c_y, c_z)                    |
| depth         | N,1   | Float euclidean distance of object to cam in meters                            |
| skeletons_2d  | N,K,2 | The 2d position of the K joints                                                |
| skeletons_3d  | N,K,3 | The 3d position of the K joints                                                |


## Frame Based Data (per image)

h,w represents height and width of the image.
The shape of these annotations is independant of the number of objects in a scene.

| Name             | Shape | Description                                                                 |
|------------------|-------|-----------------------------------------------------------------------------|
| projection       | 4,3   | Projection Matrix according to the opencv standard                          |
| image            | h,w,3 | The image in RGB format channel last (you can change that in your model)    |
| scan             | P,3   | Pointcloud containing P points from a lidarscan (x,y,z)                     |
| transform_x_to_y | 4,4   | The Rt Matrix to go from cosy X to cosy Y                                   |
| semantic_mask    | h,w,1 | Each pixel has the class_id of what is visible                              |
| instance_mask    | h,w,1 | Each pixel has the instance_id of what is visible                           |
| depth_image      | h,w,1 | Float encoding of euclidean distance of a pixel to the camera in meters     |


## Coordinate System Conventions

Following the conventions of ISO8855 and ROS makes things easier and predictable.
This means following these conventions for the coordinate systems (all right handed).

| Name         | X-Axis  | Y-Axis | Z-Axis  | Description                                               |
|--------------|---------|--------|---------|-----------------------------------------------------------|
| Image Sensor | right   | down   | forward | only pixel stuff, use for projection                      |
| Ego (Cam 0)  | forward | left   | up      | share origin with Image Sensor (3d stuff)                 |
| 3D Sensor    | forward | left   | up      | 3D Data (e.g. LiDAR) follow                               |
| Vehicle      | forward | left   | up      | Center of the vehicle the sensors are attached to         |
| World        | east    | north  | up      | Or starting position of vehicle/robot (forward, left, up) |

With these conventions switching from dataset A to dataset B should be as easy as changing one line of
code where you instantiate the dataset.
"""
from typing import Any, Dict, Iterator, List
from torch.utils.data import IterableDataset as _IterableDataset
from torch.utils.data import Dataset as _Dataset

from .parser import IParser, Parser
from .file_provider import FileProviderSequence, FileProviderIterable
from .data_promise import DataPromise


class IIterableDataset(_IterableDataset):
    """
    Interface for an iterable dataset
    (also implements the torch.utils.data.IterableDataset).

    You can use this interface when you expect a dataset in your code.
    
    If sufficient use IIterableDataset over ISequenceDataset as more datasets
    will implement with that specification as it is a subset.

    The interface requires implementations for:
    * `__iter__`
    * `__next__`
    """
    def __next__(self) -> Any:
        raise NotImplementedError("Must be implemented by subclass.")

    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError("Must be implemented by subclass.")


class ISequenceDataset(_Dataset):
    """
    Interface for a sequence dataset
    (also implements the torch.utils.data.Dataset).

    You can use this interface when you expect a dataset in your code.

    If sufficient use IIterableDataset over ISequenceDataset as more datasets
    will implement with that specification as it is a subset.

    The interface requires implementations for:
     * `__len__`
     * `__getitem__`
     * `__iter__`
     * `__next__`
    """
    def __next__(self) -> Any:
        raise NotImplementedError("Must be implemented by subclass.")

    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError("Must be implemented by subclass.")

    def __getitem__(self, index) -> Any:
        raise NotImplementedError("Must be implemented by subclass.")

    def __len__(self) -> int:
        raise NotImplementedError("Must be implemented by subclass.")


class CommonDataset(object):
    def __init__(self, file_provider_iterable: FileProviderIterable, parser: IParser, transformers=[], test_mode=False) -> None:
        """
        A common base implementation from which all datasets inherit.
        """
        super().__init__()
        self._file_provider = file_provider_iterable
        self._fp_iterator = None
        self._parser = parser
        self.transformers = []
        for transformer in transformers:
            self.transformers.append(transformer(test_mode=test_mode))

    def _process(self, sample: Dict[str, DataPromise]) -> Any:
        sample = self._parser(sample)
        return self.preprocess(sample)

    def preprocess(self, sample: Any) -> Any:
        """
        Preprocesses samples.

        The default implementation simply applies the transformers in order.
        This function can be used for transforming the data representation as well as for data augmentation.
        You can even overwrite this function to implement your own preprocessing from scratch.

        :param sample: A sample as provided by the parser (what your dataset returns if no preprocess or transformers are provided).
        :return: A sample in the format as the algorithm needs it.
        """
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


class IterableDataset(CommonDataset, IIterableDataset):
    def __init__(self, file_provider_iterable: FileProviderIterable, parser: IParser, transformers=[], test_mode=False) -> None:
        """
        An implementation of the IIterableDataset using fileprovider and parser.

        This should be used when using WebDatasets or streamed datasets.
        With this dataset random access is not possible and it can only be read in order.
        Thus the file provider is a stream (iterable).

        Do not inherit from this with your dataset implementation, provide a file provider and a
        parser or consider using and inheriting from the SimpleDataset.

        :param file_provider_iterable: The iterable file provider providing samples to the parser.
        :param parser: The parser converting samples into a usable format.
        :transformers: Transformers that are applied on the dataset to convert the format to what the model requires. (Default: [])
        :test_mode: A parameter that is passed to the constructor of the transformers (Default: False).
        """
        super().__init__(file_provider_iterable, parser, transformers=transformers, test_mode=test_mode)


class SequenceDataset(CommonDataset, ISequenceDataset):
    def __init__(self, file_provider_sequence: FileProviderSequence, parser: IParser, transformers=[], test_mode=False) -> None:
        """
        An implementation of the ISequenceDataset using fileprovider and parser.

        This should be used when using regurlar file based datasets.
        Random access is possible and might be used by a dataloader.
        Thus to enable random access the file provider is a sequence, allowing access at any index.

        Do not inherit from this with your dataset implementation, provide a file provider and a
        parser or consider using and inheriting from the SimpleDataset.

        :param file_provider_sequence: The sequence file provider providing samples to the parser.
        :param parser: The parser converting samples into a usable format.
        :transformers: Transformers that are applied on the dataset to convert the format to what the model requires. (Default: [])
        :test_mode: A parameter that is passed to the constructor of the transformers (Default: False).
        """
        super().__init__(file_provider_sequence, parser, transformers=transformers, test_mode=test_mode)
        self._file_provider = file_provider_sequence

    def __getitem__(self, index) -> Any:
        sample = self._file_provider[index]
        return self._process(sample)

class SimpleDataset(Parser, SequenceDataset):
    def __init__(self, InputType, OutputType, ignore_file_not_found=False, transformers=[], test_mode=False) -> None:
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
        :param ignore_file_not_found: If a file is missing return None instead of an exception.  (Default: False).
        :transformers: Transformers that are applied on the dataset to convert the format to what the model requires. (Default: [])
        :test_mode: A parameter that is passed to the constructor of the transformers (Default: False).
        """
        Parser.__init__(self, InputType, OutputType, ignore_file_not_found=ignore_file_not_found)
        SequenceDataset.__init__(self, [], self, transformers=transformers, test_mode=test_mode)

    def set_sample_tokens(self, sample_tokens: List[Any]) -> None:
        """
        Set the list of sample tokens.

        :param sample_tokens: A list of all sample tokens of the dataset.
        """
        self._file_provider = list(sample_tokens)
