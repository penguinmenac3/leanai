import unittest
from leanai.core.annotations import RunOnlyOnce

class TestRunOnlyOnce(unittest.TestCase):
    def setUp(self) -> None:
        self.var = 0

    def runner(self):
        return self.var

    @RunOnlyOnce
    def run_once(self):
        return self.var

    def test_output(self):
        self.var = 1
        self.assertEquals(self.runner(), self.var)
        self.var = 2
        self.assertEquals(self.runner(), self.var)

        self.var = 42
        self.assertEquals(self.run_once(), 42)
        self.var = 3
        self.assertEquals(self.run_once(), 42)


if __name__ == "__main__":
    unittest.main()
