"""doc
# leanai.data.dataset

> A generic implementation for a dataset based on parsers and file providers.

Leanai Datasets generally work by you providing an input and an output type and the implementation of the
dataset filles these fields using getters and parsers.

A simple example for 2d object detection using a LeanaiDataset would look like this.
```python
# Define Types
InputType = namedtuple("InputType", ["image"])
OutputType = namedtuple("OutputType", ["class_ids", "boxes_2d"])

# Instantiate Dataset
dataset = MyLeanaiDataset(..., InputType, OutputType, ...)

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
from typing import Any, List
from torch.utils.data import Dataset

from .transform import Transform


class LeanaiDataset(Dataset):
    def __init__(self, InputType, OutputType, transforms: List[Transform]=[]) -> None:
        """
        LeanaiDataset decodes the samples required to populate the input and output type automatically.
        
        The LeanaiDataset should never be instantiated directly.
        You should inherit it and then instantiate the inherited class.
        
        The LeanaiDataset automatically only gets those examples which are used in the InputType and OutputType.
        Data which is not used in any of the two will not be loaded thus speeding up the parsing process.
        
        To achieve this, the loader tries to call functions called "get_{attribute_name}",
        where "attribute_name" is the field name in the named tuple.
        
        For example providing this named tuple:
        `InputType = NamedTuple("InputType", image=np.ndarray)`
        will result in a call to `get_image` or an error if that function does not exist.
        The following signature will be expected:
        ```
        def get_image(self, sample_token) -> np.ndarray:
        ```
        If that is not the case, your code may break somewhere.
        
        **Arguments**
        :param InputType: A definition of a named tuple that defines the input of the neural network.
        :param OutputType: A definition of a named tuple that defines the output of the neural network.
        :param transforms: Transforms that are applied on the dataset to convert the format to what the model requires. (Default: [])
        """
        super().__init__()
        self.InputType = InputType
        self.OutputType = OutputType
        self.sample_tokens = []
        self.transforms = transforms

    def set_sample_tokens(self, sample_tokens: List[Any]) -> None:
        """
        Set the list of sample tokens.
        
        :param sample_tokens: A list of all sample tokens of the dataset.
        """
        self.sample_tokens = sample_tokens

    def __getitem__(self, index: int) -> Any:
        sample_token = self.sample_tokens[index]
        dataset_input = self._fill_type_using_getters(self.InputType, sample_token)
        dataset_output = self._fill_type_using_getters(self.OutputType, sample_token)
        return self.preprocess((dataset_input, dataset_output))

    def _fill_type_using_getters(self, namedtuple_type, sample_token):
        data = {}
        for k in namedtuple_type._fields:
            getter = getattr(self, "get_{}".format(k), None)
            if getter is not None:
                data[k] = getter(sample_token)
            else:
                raise RuntimeError(f"Missing getter (get_{k}) for dataset_input_type field: {k}")
        return namedtuple_type(**data)

    def preprocess(self, sample: Any) -> Any:
        """
        Preprocesses samples.

        The default implementation simply applies the transformers in order.
        This function can be used for transforming the data representation as well as for data augmentation.
        You can even overwrite this function to implement your own preprocessing from scratch.

        :param sample: A sample as provided by the parser (what your dataset returns if no preprocess or transformers are provided).
        :return: A sample in the format as the algorithm needs it.
        """
        for transformer in self.transforms:
            sample = transformer(sample)
        return sample
