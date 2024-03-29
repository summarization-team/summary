import os
from copy import deepcopy
from itertools import permutations
from random import seed, shuffle

import numpy as np
from nltk.metrics import masi_distance
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
                    named_entities = get_entities(original_order)
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
            original_grid = build_grid(original_order, named_entities, self.syntax)
            original_vector = create_vector(original_grid, num_sentences, num_entities, self.syntax)
            X_list.append(original_vector)
            y_list.append(1) # Proper ordering
            
            # Generate random orderings.
            orderings = self.get_grid_permutations(original_grid, num_sentences)
            
            # For each ordering, create a vector and add to dataset.
            for ordering in orderings:
                vector = create_vector(ordering, num_sentences, num_entities, self.syntax)
                X_list.append(vector)
                y_list.append(0) # Random ordering

        # Convert to NumPy ndarrays.
        X = np.vstack(X_list)
        y = np.array(y_list)
        
        return X, y

    def get_grid_permutations(self, grid, num_sentences):
        """
        Generates possible orderings of the grid.

        Args:
            grid (list of lists): The grid representing the original ordering of sentences and entities.
            num_sentences (int): The number of sentences in the grid.

        Returns:
            list: A list of grids, where each grid represents a possible reordering of the original grid.
        """
        # If num_sentences <= `threshold`, use all possible permutations.
        if num_sentences <= self.threshold:
            orderings = list(permutations(grid))
            orderings.remove(tuple(grid))
        
        # Otherwise, shuffle and select `max_permutations` possible permutations.
        else:
            orderings = []
            for i in range(self.max_permutations):
                test = self.generate_random_ordering_grid(grid)
                # Don't use a permutation that matches the original.
                # This may still result in some identical samples because there
                # is no check that an ordering doesn't already exist in `orderings`.
                # However, because we are operating on vectors rather than
                # sentences, different orderings of sentences may correspond
                # to the same grids.
                if test != grid:
                    orderings.append(test)

        return orderings

    def generate_random_ordering_grid(self, grid):
        """
        Generates a random ordering of the grid.

        Args:
            grid (list of lists): The grid representing the original ordering of sentences and entities.

        Returns:
            list of lists: A randomly shuffled grid representing a random ordering of sentences and entities.
        """
        random_ordering = deepcopy(grid)
        shuffle(random_ordering)
        return random_ordering


class InformationOrderer:
    def __init__(self, params):
        self.approach = params['approach']
        self.additional_params = params['additional_parameters']
        if self.approach == 'entity_grid':
            self.EG = EntityGrid(
                self.additional_params['training_data_path'], 
                self.additional_params['all_possible_permutations_threshold'],
                self.additional_params['max_permutations'],
                self.additional_params['syntax']
            )

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
                'max_permutations': int,
                'syntax': bool
            }
            
        Returns:
            Dictionary with updated order to the list of sentences.
        """
        syntax = additional_params['syntax']

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
                named_entities = get_entities(content[k], tokenized=True)
                num_entities = len(named_entities)
                
                # Create the grid for the ordering that is passed in.
                initial_ordering = [TreebankWordDetokenizer().detokenize(s) for s in content[k]]
                initial_grid = build_grid(initial_ordering, named_entities, syntax)
                
                # Get all possible permutations of the grid.
                full_perms = list(permutations(zip(initial_ordering, initial_grid)))
                just_sentences = [[s for s, g in o] for o in full_perms]
                just_grids = [[g for s, g in o] for o in full_perms]
                X_list = []
                for ordering in just_grids:
                    vector = create_vector(ordering, num_sentences, num_entities, syntax)
                    X_list.append(vector)

                # Convert to a NumPy array.
                X = np.array(X_list)

                # Use model to predict most likely ordering.
                probabilities = self.EG.model.predict_proba(X)[:,1]
                best_idx = np.argmax(probabilities)

                # Replace `content[k]` with the most likely ordering.
                # Ensure that this content is tokenized.
                retokenized = [word_tokenize(s) for s in just_sentences[best_idx]]
                content[k] = retokenized

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
        elif self.approach == 'baseline':
            return content
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

def get_entities(summary, tokenized=False):
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

def build_grid(sentences, entities, syntax):
    """
    If self.syntax==False, builds a binary grid indicating the presence of entities in sentences.
    If self.synatx==True, builds a grid indicating the roles of entities in sentences.

    Args:
        sentences (list of str): The sentences to be analyzed.
        entities (list of str): The entities to be searched for in the sentences.
        syntax (bool): Indicates whether to build a syntax-aware grid.

    Returns:
        list of lists: A grid where each row corresponds to a sentence and each column corresponds to an entity.
                    If syntax is True:
                        - The value at position (i, j) is 1 if the entity j is the subject of sentence i.
                        - The value is 2 if the entity j is the direct object of sentence i.
                        - The value is 3 if the entity j is any other role in sentence i.
                        - The value is 0 if the entity j is not found in sentence i.
                    If syntax is False:
                        - The value at position (i, j) is 1 if the entity j is present in sentence i, otherwise 0.
    """
    # Initialize empty array.
    array = [[0 for _ in entities] for _ in sentences]

    if syntax:
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
                if entity in sentence:
                    array[i][j] = 1
        
    return array

def create_vector(grid_list, num_sentences, num_entities, syntax):
    """
    Creates a feature vector representing the transition patterns of entities between sentences.

    Args:
        grid_list (list of lists): The grid representing the transition patterns of entities between sentences.
                                   Each row corresponds to a sentence, and each column corresponds to an entity.
                                   The values in the grid indicate the role of the entity in the sentence,
                                   if `syntax` is True.
                                   If `syntax` is False, the values indicate the presence of the entity in the sentence.
        num_sentences (int): The number of sentences in the summary.
        num_entities (int): The number of named entities in the summary.
        syntax (bool): Indicates whether the grid represents syntax-aware roles of entities in sentences.

    Returns:
        numpy.ndarray: A feature vector representing the transition patterns of entities between sentences.
                       The vector is normalized by the total number of possible transitions.
                       The length of the vector is determined by the number of unique transition patterns.
                       If `syntax` is True:
                           - The vector length is 16, representing all possible combinations of entity roles.
                       If `syntax` is False:
                           - The vector length is 4, representing the presence or absence of entities in sentences.
    """
    # Convert the grid to an array.
    grid = np.array(grid_list)

    # Use the following transition lookups.
    if syntax:
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
    return vector
