import unittest
import os
import shutil


class TestNewProject(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_create_new_project(self):
        os.system("leanai --new_project --name testproject")
        if os.path.exists("testproject"):
            self.assertTrue(os.path.exists("testproject/.vscode/launch.json"), "Cannot find launch file!")
            shutil.rmtree("testproject")
        else:
            self.assertTrue(False, "No folder testproject found.")


if __name__ == "__main__":
    unittest.main()
