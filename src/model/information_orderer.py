import os
import re
from copy import deepcopy
from itertools import permutations
from random import seed, shuffle

import numpy as np
from nltk.chunk import ne_chunk_sents
from nltk.metrics import masi_distance
from nltk.tag import pos_tag_sents
from nltk.tokenize import word_tokenize, sent_tokenize, TreebankWordDetokenizer
from nltk.tree import Tree
from sklearn.linear_model import LogisticRegression

import spacy
nlp = spacy.load('en_core_web_sm')

# Set random seed.
seed(2162024)

class EntityGrid:
    def __init__(self, training_data_filepath, threshold, max_permutations, syntax):
        self.threshold = threshold
        self.max_permutations = max_permutations
        self.syntax = syntax
        data = self.read_data(training_data_filepath)
        X, y = self.build_training_data(data)
        
        self.model = LogisticRegression()
        self.model.fit(X, y)

    def read_data(self, directory_path):
        """
        Reads gold standard summaries from files in `directory_path` and extracts relevant information.

        Args:
            directory_path (str): The path to the directory containing the files.

        Returns:
            list: A list of dictionaries, where each dictionary contains information about a text file.
                Each dictionary has the following keys:
                    - 'summary_id' (str): The filename.
                    - 'original_order' (list of str): The sentences in the original order.
                    - 'named_entities' (list of str): The named entities extracted from the text.
                    - 'num_sentences' (int): The number of sentences in the text.
                    - 'num_entities' (int): The number of named entities extracted from the text.
        """
        data = []
        for filename in os.listdir(directory_path):
            f = os.path.join(directory_path, filename)
            if os.path.isfile(f):
                with open(f, 'r', encoding='utf-8') as file:
                    # Initialize dictionary with `filename` as key.
                    summary_data = {'summary_id': filename}

                    # Break summary into sentences. Store in `summary_data`
                    original_order = sent_tokenize(file.read().strip())
                    summary_data['original_order'] = original_order

                    # Get entities and store in `summary_data`, along with the
                    # number of sentences in the summary and the number entities.
                    named_entities = self.get_entities(original_order)
                    num_sentences = len(original_order)
                    num_entities = len(named_entities)

                    summary_data['named_entities'] = named_entities
                    summary_data['num_sentences'] = num_sentences
                    summary_data['num_entities'] = num_entities

                    # Append `summary_data` to `data`.
                    data.append(summary_data)
        return data

    def build_training_data(self, data):
        """
        Builds training data from the provided summary data.

        Args:
            data (list): A list of dictionaries, where each dictionary contains information about a text file.
                Each dictionary should have the following keys:
                    - 'original_order' (list of str): The sentences in the original order.
                    - 'named_entities' (list of str): The named entities extracted from the text.
                    - 'num_sentences' (int): The number of sentences in the text.
                    - 'num_entities' (int): The number of named entities extracted from the text.

        Returns:
            tuple: A tuple containing the training data X and corresponding labels y.
                - X (numpy.ndarray): A 2D array where each row represents a vectorized version of a summary.
                - y (numpy.ndarray): A 1D array containing the labels for each summary vector in X.
                    A label of 1 indicates proper ordering, while a label of 0 indicates random ordering.
        """
        X_list = []
        y_list = []

        for summary_data in data:
            original_order = summary_data['original_order']
            named_entities = summary_data['named_entities']
            num_sentences = summary_data['num_sentences']
            num_entities = summary_data['num_entities']
                
            # Create vector for original summary and add to dataset.
            original_vector = self.create_vector(original_order, named_entities, num_sentences, num_entities)
            X_list.append(original_vector)
            y_list.append(1) # Proper ordering
            
            # Generate random orderings.
            orderings = self.get_orderings(original_order, num_sentences)

            # For each ordering, create a vector and add to dataset.
            for ordering in orderings:
                vector = self.create_vector(ordering, named_entities, num_sentences, num_entities)
                X_list.append(vector)
                y_list.append(0) # Random ordering

        # Convert to NumPy ndarrays.
        X = np.vstack(X_list)
        y = np.array(y_list)
        
        return X, y

    def get_orderings(self, sentences, num_sentences):
        """
        Generates possible orderings of sentences.

        Args:
            sentences (list of str): The sentences to be reordered.
            num_sentences (int): The number of sentences.

        Returns:
            list: A list of tuples, where each tuple represents a possible ordering of the sentences.
        """
        # If num_sentences <= `threshold``, use all possible permutations.
        if num_sentences <= self.threshold:
            orderings = list(permutations(sentences))
            orderings.remove(tuple(sentences))
        
        # Otherwise, shuffle and select `max_permutations` possible permutations.
        else:
            i = 0
            orderings = []
            while i < self.max_permutations:
                test = self.generate_random_ordering(sentences)
                # Don't use a permutation that matches the original.
                if test != sentences and test not in orderings:
                    orderings.append(test)
                    i += 1
        
        return orderings


    def generate_random_ordering(self, sentences):
        """
        Generates a random ordering of sentences.

        Args:
            sentences (list of str): The sentences to be shuffled.

        Returns:
            list of str: A randomly shuffled list of sentences.
        """
        # Create a deep copy of the list to shuffle and return.
        random_ordering = deepcopy(sentences)
        shuffle(random_ordering)
        return random_ordering
    
    def get_entities(self, summary, tokenized=False):
        """
        Extracts named entities from the given summary.

        Args:
            summary (list of str or list of list of str): The summary text. If `tokenized` is False, it should be a list of strings (sentences).
                                                        If `tokenized` is True, it should be a list of lists of strings (tokenized sentences).
            tokenized (bool, optional): Indicates whether the summary is already tokenized. Defaults to False.

        Returns:
            list of str: A list of extracted named entities.
        """

        # Tokenize sentences if necessary.
        if not tokenized:
            tokens = [word_tokenize(s) for s in summary]
        else:
            tokens = summary
        
        entities = []
        
        # Extract all named entities.
        for sentence in tokens:
            doc = spacy.tokens.Doc(nlp.vocab, words=sentence)
            doc = nlp.get_pipe('ner')(doc)
            named_entities = [ent.text for ent in doc.ents]
            entities.extend(named_entities)

        entities = list(dict.fromkeys(entities))

        # If no named entites are found, use nouns instead.
        if len(entities) == 0:
            for sentence_tokens in tokens:
                sentence = ' '.join(sentence_tokens)
                doc = nlp(sentence)
                for chunk in doc.noun_chunks:
                    if chunk.text not in entities:
                        entities.append(chunk.text)

        return entities


    def build_grid(self, sentences, entities):
        """
        If self.syntax==False, builds a binary grid indicating the presence of entities in sentences.
        If self.synatx==True, builds a grid indicating the roles of entities in sentences.

        Args:
            sentences (list of str): The sentences to be analyzed.
            entities (list of str): The entities to be searched for in the sentences.

        Returns:
            numpy.ndarray: A binary grid where each row corresponds to a sentence and each column corresponds to an entity.
                        The value at position (i, j) is 1 if the entity j is present in sentence i, otherwise 0.
        """
        # Initialize empty array.
        array = [[0 for _ in entities] for _ in sentences]

        if self.syntax:
            # Parse and iterate through sentences and entities to fill in array.
            for i, sentence in enumerate(sentences):
                for j, entity in enumerate(entities):
                    doc = nlp(sentence)
                    if entity in [ent.text for ent in doc.ents]:
                        for token in doc:
                            if token.text == entity:
                                if token.dep_ in ['nsubj', 'nsubjpass']:
                                    val = 1
                                elif token.dep_ in ['dobj']:
                                    val = 2
                                else:
                                    val = 3
                                array[i][j] = val
        else:
            # Iterate through sentences and entities to fill in array.
            for i, sentence in enumerate(sentences):
                for j, entity in enumerate(entities):
                    # check = re.search(r'\b' + entity + r'\b', sentence)
                    # if check is not None:
                        # array[i][j] = 1
                    if entity in sentence:
                        array[i][j] = 1
            
        return np.array(array)

    def create_vector(self, sentences, entities, num_sentences, num_entities):
        """
        Creates a feature vector representing the transition patterns of entities between sentences.

        Args:
            sentences (list of str): The sentences in the summary.
            entities (list of str): The named entities extracted from the summary.
            num_sentences (int): The number of sentences in the summary.
            num_entities (int): The number of named entities in the summary.

        Returns:
            numpy.ndarray: A feature vector representing the transition patterns of entities between sentences.
        """

        # Build the entity grid.
        grid = self.build_grid(sentences, entities)

        # Use the following transition lookups.
        if self.syntax:
            lookup = {
                (0, 0): 0, (0, 1): 1, (0, 2): 2, (0, 3): 3,
                (1, 0): 4, (1, 1): 5, (1, 2): 6, (1, 3): 7,
                (2, 0): 8, (2, 1): 9, (2, 2): 10, (2, 3): 11,
                (3, 0): 12, (3, 1): 13, (3, 2): 14, (3, 3): 15
            }
        else:
            lookup = {
                (0, 0): 0,
                (0, 1): 1,
                (1, 0): 2,
                (1, 1): 3
            }

        # Initialize an empty vector.
        values = [0 for _ in range(len(lookup))]

        # Calculate the total number of possible transitions.
        num_transitions = num_entities * (num_sentences - 1)
        
        # Increment each transition type in the vector.
        for j in range(num_entities):
            for i in range(num_sentences - 1):
                transition = grid[i:i+2, j]
                k = lookup[tuple(transition)]
                values[k] += 1

        # Convert the vector from counts to probabilities.
        vector = np.array(values) / num_transitions
        print(sentences)
        print(entities)
        print(vector)
        return vector



class InformationOrderer:
    def __init__(self, params):
        self.approach = params['approach']
        self.additional_params = params['additional_parameters']

    def order_content_TSP(self, content):
        """
        Orders sentences using Traveling Salesperson
        
        Args:
            Dictionary: A dictionary int the form {'file_name': [list of sentences]}
        
        Returns:
            Dictinary with updated order to the list of sentences
        """
        # Implement content ordering logic
        # Return ordered content
        for i in content:
            distances = calc_distances(content[i])
            best_route = two_opt(distances)
            new_order = []
            for x in best_route:
                new_order.append(content[i][x])
            content[i] = new_order
        return content
    
    def order_content_entity_grid(self, content, additional_params):
        """
        Orders sentences using the Entity Grid approach (Barzilay and Lapata, 2008).
        
        Args:
            content: A dictionary in the form {'file_name': [list of sentences]}
            additional_params: A dictionary in the form {
                'training_data_path': str,
                'all_possible_permutations_threshold': int,
                'max_permutations': int
            }
            
        Returns:
            Dictionary with updated order to the list of sentences.
        """
        EG = EntityGrid(
            additional_params['training_data_path'], 
            additional_params['all_possible_permutations_threshold'],
            additional_params['max_permutations'],
            additional_params['syntax']
        )
        for k in content.keys():
            num_sentences = len(content[k])

            # If there is only one sentence in `content[k]`, information
            # ordering is trivial -- continue to next summary.
            if num_sentences == 1:
                continue
        
            else:
                # Check that every item in `content[k]` is a list.
                # If it is not, it is likely an untokenized headline.
                # Tokenize it and replace the element in the list.
                for i, item in enumerate(content[k]):
                    if type(item) != list:
                        content[k][i] = word_tokenize(item)

                # Get the named entities in the content.
                named_entities = EG.get_entities(content[k], tokenized=True)
                num_entities = len(named_entities)
                
                # Get permutations.
                orderings = list(permutations(content[k]))

                # Create vector for each ordering.
                X_list = []
                for ordering in orderings:
                    sentences = [TreebankWordDetokenizer().detokenize(s) for s in ordering]
                    vector = EG.create_vector(sentences, named_entities, num_sentences, num_entities)
                    X_list.append(vector)

                # Convert to a NumPy array.
                X = np.array(X_list)

                # Use model to predict most likely ordering.
                probabilities = EG.model.predict_proba(X)[:,1]
                best_idx = np.argmax(probabilities)

                # Replace `content[k]` with the most likely ordering.
                content[k] = orderings[best_idx]

        return content

    def order_content_random(self, content):
        """
        Orders sentences randomly.
        
        Args:
            Dictionary: A dictionary in the form {'file_name': [list of sentences]}
            
        Returns:
            Dictionary with updated order to the list of sentences.
        """
        for k in content.keys():
            shuffle(content[k])
        return content

    def order_content(self, content):
        """Orders content based on the specified approach."""
        if self.approach == 'TSP':
            return self.order_content_TSP(content)
        elif self.approach == 'entity_grid':
            return self.order_content_entity_grid(content, self.additional_params)
        elif self.approach == 'random':
            return self.order_content_random(content)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")



def calc_distances(content):
    """
    Calculates the MASI distance between each sentence.

    Args:
        text (list): The list of sentences.

    Returns:
        list of list of float: A 2D list containing the distance matrix between each sentence
    """
    sentences = []
    for s in content:
        sentences.append(set(s))
        
    distances = [ [0]*len(sentences) for i in range(len(sentences))]
    
    for i in range(0, len(sentences)):
        for j in range(0, len(sentences)):
            distances[i][j] = masi_distance(sentences[i], sentences[j])
    return distances

# Reverse the order of all elements from element i to element k in array r.
two_opt_swap = lambda r,i,k: np.concatenate((r[0:i],r[k:-len(r)+i-1:-1],r[k+1:len(r)]))

def two_opt(distances):
    """
    Utilizes the two-opt algorithm to find the shortest path between sentences.

    Args:
        text (2D list): The distance matrix containing the distances between each sentence.

    Returns:
        list of int: A list of indexes containing the order of sentences that makes up the shortest path between them.
    """
    route = np.arange(len(distances))
    best_distance = path_distance(route, distances)
    for swap_first in range(0, len(route)-2):
        for swap_last in range(swap_first+1, len(route)):
            new_route = two_opt_swap(route, swap_first, swap_last)
            new_distance = path_distance(new_route, distances)
            if new_distance < best_distance:
                best_distance = new_distance
                route = new_route
    return route
    
def path_distance(route, distances):
    """
    Calculates the distance through a specific path of sentences.

    Args:
         list of ints, 2D list of floats: The list of indexes that indicate the path through the sentences, the distance matrix that contains the distance between each sentence.

    Returns:
        float: A distance through the path of sentences.
    """
    result = 0
    for c in range(1, len(route)):
        result += distances[route[c-1]][route[c]]
    return result
