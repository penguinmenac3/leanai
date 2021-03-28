import unittest


class TestTemplate(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_my_name(self):
        self.assertEquals(42, 42)


if __name__ == "__main__":
    unittest.main()
