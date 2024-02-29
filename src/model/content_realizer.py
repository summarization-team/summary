"""
This module defines classes for realizing content in different ways.

It includes an abstract base class for realization methods and concrete implementations
for simple joining of tokenized sentences and a placeholder for sentence compression.
"""

import os
import string
import re
import ast
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.tokenize import sent_tokenize
from abc import ABC, abstractmethod
import torch
from transformers import pipeline, AutoTokenizer

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  #


def set_device():
    """
    Determines and sets the computing device for tensor operations.

    If a CUDA-capable GPU is available, it sets the device to the GPU (device 0).
    Otherwise, it falls back to the CPU.

    Returns:
        torch.device: The device (GPU or CPU) where tensor operations will be performed.
    """
    return torch.device(0 if torch.cuda.is_available() else "cpu")


def is_punctuation(word):
    """
    Checks if the given word consists entirely of punctuation characters.

    Args:
        word (str): The word to be checked.

    Returns:
        bool: True if the word consists only of punctuation characters, False otherwise.
    """
    return all(char in string.punctuation for char in word)


def clean_string(input_string):
    """
    Cleans the input string by applying specific substitutions.

    This function performs the following operations:
    - Replaces occurrences of the pattern '"\'s"' with "'s".
    - Replaces occurrences of the pattern '"\'\'"' with "''".

    Args:
        input_string (str): The string to be cleaned.

    Returns:
        str: The cleaned string after applying all substitutions.
    """
    cleaned_string = re.sub(r'"\'s"', "'s", input_string)
    cleaned_string = re.sub(r'"\'\'"', "''", cleaned_string)
    return cleaned_string


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    response = openai.ChatCompletion.create(**kwargs)
    return response.choices[0].message["content"]


def get_realization_info(realization_config):
    """
    Retrieves the realization method based on the provided configuration.

    The function supports multiple realization strategies, returning an instance
    of the corresponding realization method class based on the 'method' specified
    in the configuration. It supports 'simple' and 'advanced' methods as of now.

    Args:
        realization_config (dict): A dictionary containing the realization configuration,
                                   which includes the 'method' and 'additional_parameters'.

    Returns:
        An instance of a RealizationMethod subclass corresponding to the specified method.

    Raises:
        ValueError: If the specified method is not supported.
    """
    if realization_config['method'] == 'simple':
        return SimpleJoinMethod(additional_parameters=realization_config['additional_parameters'])
    elif realization_config['method'] == 'advanced':
        return AdvancedRealizationMethod(additional_parameters=realization_config['additional_parameters'])
    elif realization_config['method'] == 'generative':
        return GenerativeRealizationMethod(additional_parameters=realization_config['additional_parameters'])
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
        """Compresses the given content using the specified compression method.

        This static method supports neural network-based compression using a specified transformers pipeline. It compresses the input content to a desired length, controlling the compression process through various optional parameters.

        Args:
            content (str): The text content to be compressed.
            method (str): The compression method to use. Currently, only 'neural' is implemented.
            transformers_pipeline (callable, optional): The transformers pipeline function to use for compression. This parameter is required if the method is 'neural'.
            **kwargs: Arbitrary keyword arguments providing additional parameters to the transformers pipeline. Common parameters include 'min_length' and 'max_length' for controlling the output length, and 'do_sample' to determine whether sampling is used during compression.

        Returns:
            str: The compressed version of the input content.

        Raises:
            NotImplementedError: If the specified compression method is not supported or implemented.

        """
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
    """
    Initializes and implements an advanced realization method for content compression and summarization.

    This class extends the `RealizationMethod` abstract base class, providing an advanced realization method that involves content compression using a specified NLP model. It initializes the necessary components for the compression task, including a tokenizer and a compression pipeline, based on the provided model identifier and additional parameters.

    Attributes:
        additional_parameters (dict): A dictionary of parameters required for initializing the tokenizer and the compression pipeline. Expected keys include 'model_id' for the model identifier, 'compression_method' for specifying the compression technique, 'min_length' and 'max_length' for defining the compression output length, and 'do_sample' to control sampling behavior during compression.

    Methods:
        realize(content): Compresses and summarizes the input content. It first detokenizes the input sentences, then compresses the combined text using the specified compression method and parameters, and finally tokenizes the compressed string into sentences.

    Args:
        additional_parameters (dict): Configuration parameters for model initialization and content compression. Must include 'model_id' to specify the pre-trained model for tokenization and compression, and may include 'compression_method', 'min_length', 'max_length', and 'do_sample' to customize the compression behavior.

    Raises:
        NotImplementedError: If the specified compression method in `additional_parameters` is not supported.

    """

    def __init__(self, additional_parameters):
        super().__init__(additional_parameters)
        device = set_device()
        self.tokenizer = AutoTokenizer.from_pretrained(additional_parameters['model_id'], model_max_length=512)
        self.compression_pipeline = pipeline(
            task="summarization",
            model=additional_parameters['model_id'],
            tokenizer=self.tokenizer,
            device=device
        )

    def realize(self, content):
        """Compresses and realizes the provided content into a concise form using a pre-defined compression pipeline.

        This method takes a dictionary of tokenized sentences, detokenizes them to form a coherent paragraph, and then compresses the paragraph using a neural network-based compression method specified in the class' initialization parameters. The compression process is tailored by parameters such as the minimum and maximum length of the output, and whether or not to use sampling, as dictated by the `additional_parameters` provided during class initialization.

        The method effectively reduces the length of the input content while aiming to retain the most important information, making it suitable for applications requiring concise summaries of larger text bodies.

        Args:
            content (dict): A dictionary where each key maps to a list of tokenized sentences. Each sentence is represented as a list of strings (tokens).

        Returns:
            list: A list of strings, where each string is a sentence from the compressed and summarized content. The content is first combined and compressed into a single string, which is then tokenized back into sentences.

        Example:
            Given a dictionary of tokenized sentences, `realize` will compress these into a shorter, summarized form:

            ```
            content = {
                "paragraph1": [["This", "is", "a", "sentence", "."], ["This", "is", "another", "sentence", "."]],
                "paragraph2": [["Yet", "another", "sentence", "."]]
            }
            realized_content = advancedRealizationMethodInstance.realize(content)
            ```

            `realized_content` will then contain the compressed form of the input, split into individual sentences.
        """
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


class GenerativeRealizationMethod(RealizationMethod):
    """
    A realization method that uses OpenAI's GPT model for generating content, with the engine specified
    as an additional parameter. This allows for the dynamic selection of the GPT engine (e.g., GPT-3.5-turbo).

    This method leverages the generative capabilities of a specified GPT model to realize content based on
    tokenized input sentences. It transforms the list of tokenized sentences into a prompt for the model,
    and uses the model's response as the realized content.

    Attributes:
        additional_parameters (dict): Configuration parameters for the GPT model, including the engine name,
                                      settings for the generation process such as temperature, max tokens, etc.,
                                      and the API key sourced from an environment variable.
    """

    def __init__(self, additional_parameters):
        """
        Initializes the GenerativeRealizationMethod method with specific parameters for the GPT model.

        Args:
            additional_parameters (dict): A dictionary containing parameters for content generation,
                                          including the model engine name and settings for the generation process.
                                          The API key is sourced from the OPENAI_API_KEY environment variable.
        """
        super().__init__(additional_parameters)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model_id = additional_parameters.get('model_id', 'gpt-3.5-turbo')

    def realize(self, content):
        """
        Realizes the given content using the specified GPT model.

        Args:
            content (dict): A dictionary where each key maps to a list of tokenized sentences,
                            which will be used to construct the prompt for the model.

        Returns:
            str: The generated content as a single string, based on the model's response to the prompt.
        """
        detokenizer = TreebankWordDetokenizer()
        prompt_sentences = [s for s_list in content.values() for s in s_list]
        prompt_sentences = detokenizer.detokenize([' '.join(sentence) for sentence in prompt_sentences])

        response = self._get_completion(
            concatenated_sentences=prompt_sentences,
            model=self.model_id,
            temperature=self.additional_parameters.get('temperature', 0.0),
            n=self.additional_parameters.get('temperature', 1),
            max_tokens=round(self.additional_parameters.get('max_length', 130)*1.4)
        )

        response_tokenized = sent_tokenize(response)
        return response

    def _get_completion(self, concatenated_sentences, model, temperature, n, max_tokens):
        prompt = f"""
        Your task is mutli-document summarizaiton. Specifically, you will be addressing the "content realization" \
        step of an extractive summarization system. You will be provided a set of sentences \
        that have been extracted from news articles pertaining to a particular topic or event. \
        These sentences have also been ordered. You should return a narrative string. \
        First, review the sentences to ensure that the flow coherently. Second, enhance readability. \
        Third, compress or fuse the sentences as necessary. Your output should be no more than 100 words.

        Selected Content: '''{concatenated_sentences}'''
        """
        messages = [{"role": "user", "content": prompt}]
        completion = completion_with_backoff(model=model, messages=messages, temperature=temperature, n=n, max_tokens=max_tokens)
        return completion


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
