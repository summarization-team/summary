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
        self.assertIsInstance(realized_content, str)
        self.assertIn("This is a test.", realized_content)
        self.assertIn("Another sentence.", realized_content)

    def test_content_realizer_with_simple_join(self):
        method = SimpleJoinMethod(additional_parameters={})
        realizer = ContentRealizer(method=method)
        content = {"sentences": [["This", "is", "simple", "join", "."]]}
        realized_content = realizer.realize_content(content)
        self.assertIn("This is simple join.", realized_content)

    def test_get_realization_info(self):
        config = {'method': 'simple', 'additional_parameters': {}}
        method = get_realization_info(config)
        self.assertIsInstance(method, SimpleJoinMethod)

    # Updated Sentence Compression Tests to use model_id from config
    def setUp(self):
        # Updated to include model_id in the configuration
        self.compression_method = SentenceCompressionMethod(
            additional_parameters= {
                'model_id': 't5-base',
                "max_length": 30,
                "min_length": 5,
                "do_sample": False
                                   }
        )

    def test_compression_single_sentence(self):
        content = {"sentences": [['HONG', 'KONG', 'December', '8', 'Xinhua', 'Hong', 'Kong', 'health', 'authorities', 'are', 'stepping', 'up', 'the', 'hunt', 'for', 'the', 'source', 'of', 'a', 'deadly', 'influenza', 'strain', 'previously', 'found', 'only', 'in', 'birds', 'which', 'has', 'caused', 'a', 'global', 'alert' , '.']]}
        compressed_content = self.compression_method.realize(content)
        print(compressed_content)
        self.assertIsInstance(compressed_content, str)
        self.assertTrue(len(compressed_content) < len(' '.join(content["sentences"][0])))
        self.assertNotEqual(compressed_content, '')

    def test_compression_multiple_sentences(self):
        content = {
            "sentences": [
                ["This", "is", "the", "first", "long", "sentence", "that", "needs", "compression", "."],
                ["Here", "is", "another", "sentence", "that", "also", "requires", "compression", "."]
            ]
        }
        compressed_content = self.compression_method.realize(content)
        self.assertIsInstance(compressed_content, str)
        original_content_length = sum(len(' '.join(sentence)) for sentence in content["sentences"])
        self.assertTrue(len(compressed_content) < original_content_length)
        self.assertNotEqual(compressed_content, '')

    def test_compression_with_empty_input(self):
        content = {"sentences": []}
        compressed_content = self.compression_method.realize(content)
        self.assertEqual(compressed_content, '')

if __name__ == '__main__':
    unittest.main()
