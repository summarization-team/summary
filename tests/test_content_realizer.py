import unittest
from src.model.content_realizer import (
    SimpleJoinMethod,
    SentenceCompressionMethod,
    is_punctuation,
    clean_string,
    ContentRealizer,
    get_realization_info
)


class TestContentRealization(unittest.TestCase):
    def test_is_punctuation(self):
        self.assertTrue(is_punctuation("!!!"))
        self.assertFalse(is_punctuation("word!"))

    def test_clean_string(self):
        self.assertEqual(clean_string("'s"), "'s")
        self.assertEqual(clean_string("''"), "''")

    def test_simple_join_method(self):
        method = SimpleJoinMethod(additional_parameters={})
        content = {"sentences": [["This", "is", "a", "test", "."], ["Another", "sentence", "."]]}
        realized_content = method.realize(content)
        self.assertEqual(realized_content, ["This is a test.", "Another sentence."])

    def test_sentence_compression_method_not_implemented(self):
        method = SentenceCompressionMethod(additional_parameters={})
        content = {"sentences": [["This", "sentence", "will", "not", "be", "compressed", "."]]}
        with self.assertRaises(NotImplementedError):
            method.realize(content)

    def test_content_realizer_with_simple_join(self):
        method = SimpleJoinMethod(additional_parameters={})
        realizer = ContentRealizer(method=method)
        content = {"sentences": [["This", "is", "simple", "join", "."]]}
        realized_content = realizer.realize_content(content)
        self.assertEqual(realized_content, ["This is simple join."])

    def test_get_realization_info(self):
        config = {'method': 'simple', 'additional_parameters': {}}
        method = get_realization_info(config)
        self.assertIsInstance(method, SimpleJoinMethod)


if __name__ == '__main__':
    unittest.main()
