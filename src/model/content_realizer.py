"""
This module defines classes for realizing content in different ways.

It includes an abstract base class for realization methods and concrete implementations
for simple joining of tokenized sentences and a placeholder for sentence compression.
"""

import string
from nltk.tokenize.treebank import TreebankWordDetokenizer
from abc import ABC, abstractmethod


def is_punctuation(word):
    return all(char in string.punctuation for char in word)


def get_realization_info(realization_config):
    if realization_config['method'] == 'simple':
        return SimpleJoinMethod(additional_parameters=realization_config['additional_parameters'])
    elif realization_config['method'] == 'sentence_compression':
        return SentenceCompressionMethod(additional_parameters=realization_config['additional_parameters'])
    else:
        raise ValueError(f"Unknown realization strategy: {realization_config['method']}")


class RealizationMethod(ABC):
    """
    Abstract base class for realization methods.

    This class defines a contract for realization methods which take a list of tokenized sentences
    and return a string representation.
    """

    def __init__(self, additional_parameters):
        self.additional_parameters = additional_parameters

    @abstractmethod
    def realize(self, content):
        """Realizes the given content into a string."""
        pass

    @staticmethod
    def _truncate_sentences(content, max_token_length):
        """
        Truncate a list of sentences to meet a specified maximum word count.

        This function takes a list of sentences, where each sentence is represented as a list of words.
        Starting with the first sentence in the list, it selects consecutive sentences until the total
        word count reaches or exceeds the specified maximum word count.

        Args:
            content (list of list of str): A list of sentences, where each sentence is a list of words.
            max_token_length (int): The maximum word count to which the sentences should be truncated.

        Returns:
            list of list of str: A truncated list of sentences that collectively have a word count less
            than or equal to the specified maximum word count.
        """
        truncated_content = []  # Initialize an empty list of sentences
        total_word_count = 0

        for sentence in content:
            sentence_word_count = 0

            for word in sentence:
                if not is_punctuation(word):
                    sentence_word_count += 1

            if total_word_count + sentence_word_count <= max_token_length:
                truncated_content.append(sentence)
                total_word_count += sentence_word_count
            else:
                break  # Stop adding sentences when the word limit is reached

        return truncated_content


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
        if "max_token_length" in self.additional_parameters:
            sentences = self._truncate_sentences(content=content.values(),
                                                 max_token_length=self.additional_parameters['max_token_length'])
        else:
            sentences = content

        detokenizer = TreebankWordDetokenizer()
        detokenized_sentences = [detokenizer.detokenize(sentence) for sentence in sentences]

        summary = ' '.join(detokenized_sentences)

        return summary


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
