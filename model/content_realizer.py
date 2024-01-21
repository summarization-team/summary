"""
This module defines classes for realizing content in different ways.

It includes an abstract base class for realization methods and concrete implementations
for simple joining of tokenized sentences and a placeholder for sentence compression.
"""

from nltk.tokenize.treebank import TreebankWordDetokenizer
from abc import ABC, abstractmethod

# # Ensure necessary NLTK data is downloaded
# nltk.download('punkt')

class RealizationMethod(ABC):
    """
    Abstract base class for realization methods.

    This class defines a contract for realization methods which take a list of tokenized sentences
    and return a string representation.
    """

    @abstractmethod
    def realize(self, content):
        """Realizes the given content into a string."""
        pass

class SimpleJoinMethod(RealizationMethod):
    """
    A realization method that simply joins tokenized sentences.

    This method uses the NLTK TreebankWordDetokenizer for joining tokens, which handles
    spaces and punctuation in a way typical for English text.
    """

    def realize(self, content):
        """Realizes content by joining tokenized sentences.

        Args:
            content (list of list of str): A list of sentences, where each sentence is represented as a list of tokens.

        Returns:
            str: The realized content as a single string.
        """
        detokenizer = TreebankWordDetokenizer()
        sentences = [detokenizer.detokenize(sentence) for sentence in content]
        return ' '.join(sentences)

class SentenceCompressionMethod(RealizationMethod):
    """
    Placeholder for a future sentence compression realization method.

    This method is intended to implement advanced sentence compression techniques and
    is currently not implemented.
    """

    def realize(self, content):
        """Realizes content using sentence compression.

        Args:
            content (list of list of str): A list of sentences, where each sentence is represented as a list of tokens.

        Raises:
            NotImplementedError: Indicates that the method is not yet implemented.
        """
        raise NotImplementedError("Sentence compression method is not yet implemented.")

class ContentRealizer:
    """
    A class for realizing content using specified realization methods.

    This class allows for the setting and use of different content realization methods.
    """

    def __init__(self, method: RealizationMethod):
        """
        Initializes the ContentRealizer with a specific realization method.

        Args:
            method (RealizationMethod): An instance of a realization method.
        """
        self.method = method

    def set_method(self, method: RealizationMethod):
        """
        Sets the realization method.

        Args:
            method (RealizationMethod): An instance of a realization method to be used for content realization.
        """
        self.method = method

    def realize_content(self, content):
        """
        Realizes the provided content using the current realization method.

        Args:
            content (list of list of str): The content to be realized, provided as a list of tokenized sentences.

        Returns:
            str: The realized content.
        """
        return self.method.realize(content)
