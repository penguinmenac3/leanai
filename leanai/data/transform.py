"""doc
# leanai.data.transform

> A transform converts the data from a general format into what the neural network needs.
"""


class Transform():
    def __init__(self):
        """
        A transform must implement `__call__`.
        
        A transform is a callable that gets the data (usually a tuple of feature, label), transforms it and returns the data again (usually as a tuple of feature, label).
        The last transform must output a tuple of feature of type NetworkInput (namedtuple) and label of type NetworkOutput (namedtuple) to be able to pass it to the neural network.

        Example:
        ```python
        def __call__(self, sample: Tuple[DatasetInput, DatasetOutput]) -> Tuple[NetworkInput, NetworkTargets]:
        ```
        """
        pass

    def __call__(self, *args):
        """
        This function gets the data from the previous transform or dataset as input and should output the data again.

        :param args: The input data.
        :return: The output data.
        """
        raise NotImplementedError

    @property
    def version(self):
        """
        Defines the version of the transform. The name can be also something descriptive of the method.

        :return: The version number of the transform.
        """
        raise NotImplementedError
