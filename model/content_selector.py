import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx

class ContentSelector:
    def __init__(self, approach='default'):
        self.approach = approach


    def read_data2dic(self, data_dir):
        """
        Recursively reads the content of files in the specified directory and subdirectories
        and creates a nested dictionary. The outer dictionary has keys as parent directory paths,
        and values are dictionaries representing files within each directory. The inner dictionaries
        have keys as file paths and values as lists of unique sentences extracted from the files.

        Parameters:
        - data_dir (str): The path to the directory containing the data files.

        Returns:
        dict: A nested dictionary where keys are parent directory paths, values are dictionaries 
            with keys as file paths and values as lists of unique sentences extracted from 
            corresponding files.
        """
        parent_dict = {}
        for root, files in os.walk(data_dir):
            file_dict = {}
            for file_name in files:
                file_path = os.path.join(root, file_name)

                if os.path.isfile(file_path):
                    with open(file_path, 'r') as file:

                        file_content = file.readlines()
                        sentlist = []
                        for line in file_content:

                            # if line in list format, it's a sentence that we want to analyze
                            if line[0] == "[":
                                line = line.strip("[]\n")
                                wordlist = line.split(", ")
                                wordlist = [word[1:-1] for word in wordlist]
                                sentence = " ".join(wordlist)

                                # ensure each sentence is unique so that we don't have repetition
                                if sentence not in sentlist:
                                    sentlist.append(sentence)
                            else:
                                pass

                    file_dict[file_path] = sentlist
            
            parent_dict[root] = file_dict

        return parent_dict



    def select_content(self, docset):
        if self.approach == 'approach1':
            return self._select_content_approach1(docset, 5)
        elif self.approach == 'approach2':
            return self._select_content_approach2(docset)
        else:
            return self._select_content_default(docset)



    def _select_content_approach1(self, docset, num_sentences):
        """
        Selects top sentences from a document set using a content selection approach.
        The approach involves creating sentence vectors using CountVectorizer, computing
        cosine similarity between sentences, and using the TextRank algorithm (through
        PageRank on a similarity graph) to identify top sentences.

        Parameters:
        - docset (dict): A nested dictionary where keys are parent directory paths, and
        values are dictionaries with keys as file paths and values as lists of sentences
        extracted from corresponding files.
        - num_sentences (int): The number of top sentences to select from each file.

        Returns:
        dict: A dictionary where keys are parent directory paths, and values are lists
            of dictionaries, each containing 'file_name' and 'file_top_sent' keys.
            'file_name' represents the file's name, and 'file_top_sent' is a list of
            top sentences selected for that file.
        """
        selected_sent = {}

        for parent in docset.keys():
            for filename in docset.keys():

                sentlist = docset[filename]
                
                # create vectors for each sentence, use cosine similarity to compare them
                vectorizer = CountVectorizer(stop_words="english")
                sentence_vectors = vectorizer.fit_transform(sentlist)
                similarity_matrix = cosine_similarity(sentence_vectors)

                # graph the resulting similarity matrix, then use the TextRank algorithm (thru PageRank) to find top sentence scores
                graph = nx.from_numpy_array(similarity_matrix)
                sentence_scores = nx.pagerank(graph, alpha = 0.85, max_iter = 100) 
                
                ranked_sentindices = sorted(range(len(sentence_scores)), key=lambda index: sentence_scores[index], reverse=True)
                top_sentindices = ranked_sentindices[:num_sentences]

                top_sentences = [sentlist[i] for i in top_sentindices]

                # Store the content in the dictionary based on the parent directory
                parent_directory = parent
                if parent_directory not in selected_sent.keys():
                    selected_sent[parent_directory] = []

                
                selected_sent[parent_directory].append({
                    'file_name': filename,
                    'file_top_sent': top_sentences
                })

        return selected_sent



    def _select_content_approach2(self, docset):
        # Implement the second content selection approach
        # Return selected content
        pass

    def _select_content_default(self, docset):
        # Implement a default content selection approach
        # Return selected content
        pass
