"""
This module defines classes for realizing content in different ways.

It includes an abstract base class for realization methods and concrete implementations
for simple joining of tokenized sentences and a placeholder for sentence compression.
"""

import string
import re
import ast
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize
from abc import ABC, abstractmethod
import torch
from transformers import pipeline, AutoTokenizer


def get_device():
    """
    Checks for GPU availability and returns the appropriate device ('cuda' or 'cpu').

    Returns:
        str: The device type. 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        # If a GPU is available, return 'cuda' to use it
        return 0
    else:
        # If no GPU is available, default to using the CPU
        return 1

def is_punctuation(word):
    return all(char in string.punctuation for char in word)


def clean_string(input_string):
    # Replace 's
    cleaned_string = re.sub(r'"\'s"', "'s", input_string)
    # Replace "''"
    cleaned_string = re.sub(r'"\'\'"', "''", cleaned_string)
    return cleaned_string


def get_realization_info(realization_config):
    if realization_config['method'] == 'simple':
        return SimpleJoinMethod(additional_parameters=realization_config['additional_parameters'])
    elif realization_config['method'] == 'advanced':
        return AdvancedRealizationMethod(additional_parameters=realization_config['additional_parameters'])
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
    def _truncate_sentences(content, max_length):
        """
        Truncate a list of sentences to meet a specified maximum word count.

        This function takes a list of sentences, where each sentence is represented as a list of words.
        Starting with the first sentence in the list, it selects consecutive sentences until the total
        word count reaches or exceeds the specified maximum word count.

        Args:
            content (list of list of str): A list of sentencesd, where each sentence is a list of words.
            max_length (int): The maximum word count to which the sentences should be truncated.

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

            if total_word_count + sentence_word_count <= max_length:
                truncated_content.append(sentence)
                total_word_count += sentence_word_count
            else:
                break  # Stop adding sentences when the word limit is reached

        return truncated_content

    @staticmethod
    def _compress(content, method, transformers_pipeline=None, **kwargs):
        if method == 'neural':
            compressed_content = transformers_pipeline(
                content,
                min_length=kwargs.get('min_length', 5),
                max_length=kwargs.get('max_length', 100),
                do_sample=kwargs.get('do_sample', False)
            )[0]['summary_text']
        else:
            return NotImplementedError("neural compression is the only implemented method for compression")

        return compressed_content

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
        sentences = [s for s_list in content.values() for s in s_list]

        if "max_length" in self.additional_parameters:
            sentences = self._truncate_sentences(content=sentences,
                                                 max_length=self.additional_parameters['max_length'])

        sentences = [[clean_string(token) for token in sentence] for sentence in sentences]
        detokenizer = TreebankWordDetokenizer()
        detokenized_sentences = [detokenizer.detokenize(sentence) for sentence in sentences]

        return detokenized_sentences


class AdvancedRealizationMethod(RealizationMethod):
    def __init__(self, additional_parameters):
        super().__init__(additional_parameters)
        device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(additional_parameters['model_id'], model_max_length=512)
        self.compression_pipeline = pipeline(
            task="summarization",
            model=additional_parameters['model_id'],
            tokenizer=self.tokenizer,
            device=device
        )

    def realize(self, content):
        detokenizer = TreebankWordDetokenizer()
        sentences = [s for s_list in content.values() for s in s_list]
        sentences = [[clean_string(token) for token in sentence] for sentence in sentences]
        detokenized_sentences = [detokenizer.detokenize(sentence) for sentence in sentences]
        sentence_str = ' '.join(detokenized_sentences)
        compressed_str = self._compress(
            content=sentence_str,
            method=self.additional_parameters['compression_method'],
            transformers_pipeline=self.compression_pipeline,
            min_length=self.additional_parameters['min_length'],
            max_length=self.additional_parameters['max_length'],
            do_sample=self.additional_parameters['do_sample']
        )
        compressed_sentences = sent_tokenize(compressed_str)
        return compressed_sentences

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
