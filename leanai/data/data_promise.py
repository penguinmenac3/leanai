"""doc
# leanai.data.data_promise

> The promise object for data can be used to abstract how the data is loaded and only load it lazy.

"""


class DataPromise(object):
    """
    The interface for a datapromise can be used when data is expected, but the user
    does not care if the data is loaded lazy or ahead of time.
    """
    @property
    def data(self) -> bytes:
        """
        Get the data as raw bytes.

        Must be implemented by all subclasses.
        """
        raise NotImplementedError("The data property must be implemented by subclasses.")


class DataPromiseFromFile(DataPromise):
    def __init__(self, filename) -> None:
        """
        A promise on the data in a file.
        Only loads the data on access and buffers it then.
        """
        self._filename = filename
        self._buffer = None

    @property
    def data(self) -> bytes:
        """
        Get the data as raw bytes.
        
        Lazy loads the data and buffers it for future access.
        """
        if self._buffer is None:
            with open(self._filename, "rb") as f:
                self._buffer = f.read()
        return self._buffer


class DataPromiseFromBytes(DataPromise):
    def __init__(self, bytes) -> None:
        """
        A promise on bytes that are already loaded.

        Can be usefull when promising data from a tar stream.
        """
        self._bytes = bytes

    @property
    def data(self) -> bytes:
        """
        Get the data as raw bytes.
        
        Uses the bytes provided in the constructor.
        """
        return self._bytes
