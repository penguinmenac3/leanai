"""doc
# leanai.data.transformer

> A transformer converts the data from a general format into what the neural network needs.
"""


class Transformer():
    def __init__(self):
        """
        A transformer must implement `__call__`.
        
        A transformer is a callable that gets the data (usually a tuple of feature, label), transforms it and returns the data again (usually as a tuple of feature, label).
        The last transformer must output a tuple of feature of type NetworkInput (namedtuple) and label of type NetworkOutput (namedtuple) for babilim to be able to pass it to the neural network.
        """
        pass

    def __call__(self, *args):
        """
        This function gets the data from the previous transformer or dataset as input and should output the data again.

        :param args: The input data.
        :return: The output data.
        """
        raise NotImplementedError

    @property
    def version(self):
        """
        Defines the version of the transformer. The name can be also something descriptive of the method.

        :return: The version number of the transformer.
        """
        raise NotImplementedError
