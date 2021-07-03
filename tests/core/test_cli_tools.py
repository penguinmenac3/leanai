import unittest
import os
import shutil


class TestCliTools(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_create_new_project(self):
        os.system("leanai --new_project --name testproject")
        self.assertTrue(os.path.exists("testproject"), "No folder testproject found.")
        self.assertTrue(os.path.exists("testproject/.vscode/launch.json"), "Cannot find launch file!")

    def tearDown(self) -> None:
        if os.path.exists("testproject"):
            shutil.rmtree("testproject")


if __name__ == "__main__":
    unittest.main()
