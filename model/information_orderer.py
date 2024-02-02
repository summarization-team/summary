from nltk.metrics import masi_distance
from nltk.tokenize import word_tokenize
import numpy as np

class InformationOrderer:
    def order_content(self, content):
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
        sentences.append(set(word_tokenize(s)))
        
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
