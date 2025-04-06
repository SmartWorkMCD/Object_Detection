import unittest
from app.functions import file_key

class TestFileKey(unittest.TestCase):
    def test_file_key_with_numeric_filenames(self):
        self.assertEqual(file_key("123.txt"), 123)
        self.assertEqual(file_key("456.log"), 456)

    def test_file_key_with_non_numeric_filenames(self):
        self.assertEqual(file_key("abc.txt"), float("inf"))
        self.assertEqual(file_key("file.log"), float("inf"))

    def test_file_key_with_mixed_filenames(self):
        self.assertEqual(file_key("789.txt"), 789)
        self.assertEqual(file_key("test123.log"), float("inf"))

    def test_file_key_with_no_extension(self):
        self.assertEqual(file_key("42"), 42)
        self.assertEqual(file_key("non_numeric"), float("inf"))

    def test_file_key_with_empty_filename(self):
        self.assertEqual(file_key(""), float("inf"))


if __name__ == "__main__":
    unittest.main()
