from nltk.metrics import masi_distance
from nltk.tokenize import word_tokenize, sent_tokenize, TreebankWordDetokenizer
from nltk.tag import pos_tag_sents
from nltk.chunk import ne_chunk_sents
from nltk.tree import Tree
from itertools import permutations
from random import seed, shuffle
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import numpy as np
import os, re

# Set filepath for training data. (Put this into config eventually.)
GOLD_TRAINING = '../data/gold/training'

# Set random seed.
seed(2162024)

class EntityGrid:
    def __init__(self, training_data_filepath):
        data = self.read_data(training_data_filepath)
        X, y = self.build_training_data(data)
        
        self.model = LogisticRegression()
        self.model.fit(X, y)

    def read_data(self, directory_path):
        """
        Return list of lists.
        Outer list represents single summary file.
        Inner list is sentences in that summary.
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
        """
        X_list = []
        y_list = []

        for summary_data in data:
            original_order = summary_data['original_order']
            named_entities = summary_data['named_entities']
            num_sentences = summary_data['num_sentences']
            num_entities = summary_data['num_entities']
                
            # Create vector for original summary and add to dataset.
            original_vector = create_vector(original_order, named_entities, num_sentences, num_entities)
            X_list.append(original_vector)
            y_list.append(1) # Proper ordering
            
            # Generate random orderings.
            orderings = self.get_orderings(original_order, num_sentences)

            # For each ordering, create a vector and add to dataset.
            for ordering in orderings:
                vector = create_vector(ordering, named_entities, num_sentences, num_entities)
                X_list.append(vector)
                y_list.append(0)

        X = np.vstack(X_list)
        y = np.array(y_list)
        
        return X, y


    def get_orderings(self, sentences, num_sentences):
        """
        """
        # If num_sentences <= 3, use all possible permutations
        if num_sentences <= 3:
            orderings = list(permutations(sentences))
            orderings.remove(tuple(sentences))
        
        # Otherwise, use shuffle and select 10 possible permutations.
        else:
            i = 0
            orderings = []
            while i < 10:
                test = self.generate_random_ordering(sentences)
                if test != sentences and test not in orderings:
                    orderings.append(test)
                    i += 1
        
        return orderings


    def generate_random_ordering(self, sentences):
        """Shuffle list of sentences and return."""
        random_ordering = deepcopy(sentences)
        shuffle(random_ordering)
        return random_ordering

class InformationOrderer:
    def __init__(self, approach='TSP'):
        self.approach = approach

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
    
    def order_content_entity_grid(self, content):
        """
        Orders sentences using the Entity Grid approach (Barzilay and Lapata, 2008).
        
        Args:
            Dictionary: A dictionary in the form {'file_name': [list of sentences]}
            
        Returns:
            Dictionary with updated order to the list of sentences.
        """
        EG = EntityGrid(GOLD_TRAINING)
        for k in content.keys():
            num_sentences = len(content[k])

            # If there is only one sentence in `content[k]`, information
            # ordering is trivial -- continue to next summary.
            if num_sentences == 1:
                continue
        
            else:
                # Check that every item in `content[k]` is a list.
                # If it is not a list, print `content[k]` and break.
                for i, item in enumerate(content[k]):
                    if type(item) != list:
                        # print(content[k])
                        # print(item)
                        content[k][i] = word_tokenize(item)
                        # print(content[k])
                # Get the named entities in the content.
                named_entities = get_entities(content[k], tokenized=True)
                num_entities = len(named_entities)
                
                # Get permutations.
                orderings = list(permutations(content[k]))

                # Create vector for each ordering.
                X_list = []
                for ordering in orderings:
                    sentences = [TreebankWordDetokenizer().detokenize(s) for s in ordering]
                    vector = create_vector(sentences, named_entities, num_sentences, num_entities)
                    X_list.append(vector)

                X = np.array(X_list)

                # Use model to predict most likely ordering.
                probabilities = EG.model.predict_proba(X)[:,1]
                best_idx = np.argmax(probabilities)
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
            return self.order_content_entity_grid(content)
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

def get_entities(summary, tokenized=False):
    """Extract and return NEs from a list of sentences."""
    if not tokenized:
        tokens = [word_tokenize(s) for s in summary]
    else:
        tokens = summary
    tags = pos_tag_sents(tokens)
    entities = ne_chunk_sents(tags, binary=True)
    NE = []
    for entity in entities:
        for subtree in entity:
            if isinstance(subtree, Tree):
                entity_name = " ".join([word for word, tag in subtree.leaves()])
                if entity_name not in NE:
                    NE.append(entity_name)

    # If no NEs are extracted, just use nouns instead. (e.g., 34, 254)
    if len(NE) == 0:
        is_noun = lambda pos: pos[:2] == 'NN'
        for tagged_sent in tags:
            for word, pos in tagged_sent:
                if is_noun(pos) and word not in NE:
                    NE.append(word)
    
    return NE

def build_grid(sentences, entities):
    """
    Construct and return entity grid as NumPy array.
    num_columns = len(entities)
    num_rows = len(sentences)
    Cell is 1 if entity present, 0 otherwise.
    """
    array = [[0 for _ in entities] for _ in sentences]
    for i, sentence in enumerate(sentences):
        for j, entity in enumerate(entities):
            check = re.search(r'\b' + entity + r'\b', sentence)
            if check is not None:
                array[i][j] = 1
    
    return np.array(array)

def create_vector(sentences, entities, num_sentences, num_entities):
    """
    From grid, create dictionary of transitions."""

    grid = build_grid(sentences, entities)

    lookup = {
        (0, 0): 0,
        (0, 1): 1,
        (1, 0): 2,
        (1, 1): 3
    }

    values = [0 for _ in range(len(lookup))]

    num_transitions = num_entities * (num_sentences - 1)
    
    for j in range(num_entities):
        for i in range(num_sentences - 1):
            transition = grid[i:i+2, j]
            k = lookup[tuple(transition)]
            values[k] += 1

    vector = np.array(values) / num_transitions

    return vector
