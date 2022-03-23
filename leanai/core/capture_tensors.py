_hierarchy = ""
_listening = []
_captured = {}


def capture(name: str, value: any) -> any:
    fully_qualified_name = f"{_hierarchy}{name}"
    if fully_qualified_name in _listening:
        _captured[fully_qualified_name] = value
    return value


def clear_captures():
    _listening.clear()
    _captured.clear()


def get_captured(name: str) -> any:
    fully_qualified_name = f"{_hierarchy}{name}"
    return _captured[fully_qualified_name]


def _start_capture(name: str):
    fully_qualified_name = f"{_hierarchy}{name}"
    _listening.append(fully_qualified_name)


def _end_capture(name: str):
    fully_qualified_name = f"{_hierarchy}{name}"
    _listening.remove(fully_qualified_name)


class CaptureNamespace(object):
    def __init__(self, namespace, record=[]):
        global _hierarchy
        self._record = record
        self._old_hierarchy = _hierarchy
        self._namespace = namespace

    def __enter__(self):
        global _hierarchy
        _hierarchy = f"{_hierarchy}{self._namespace}/"
        for x in self._record:
            _start_capture(x)

    def __exit__(self, *args):
        global _hierarchy
        for x in self._record:
            _end_capture(x)
        _hierarchy = self._old_hierarchy


def _test():
    _start_capture("foo")
    capture("foo", 42)
    assert get_captured("foo") == 42

    with CaptureNamespace("bar", record=["baz"]):
        capture("foo", 123)
        capture("baz", 33)
    assert get_captured("bar/baz") == 33
    with CaptureNamespace("bar"):
        assert get_captured("baz") == 33

    capture("foo", 1)
    with CaptureNamespace("bar"):
        capture("foo", 2)
        capture("baz", 3)
    
    assert get_captured("foo") == 1
    assert get_captured("bar/baz") == 33, "Expected 33 as it is no longer captured."
    with CaptureNamespace("bar"):
        assert get_captured("baz") == 33, "Expected 33 as it is no longer captured."
    
    try:
        get_captured("bar/foo")
        assert False
    except KeyError:
        pass
    
    clear_captures()

    try:
        get_captured("foo")
        assert False
    except KeyError:
        pass
    print("Test: passed")


if __name__ == "__main__":
    _test()
