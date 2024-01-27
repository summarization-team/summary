import os
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx


"""
This script walks through a specified parent directory, reads text files, and performs content selection
using Cosine Similarity and PageRank algorithm. It extracts sentences from the text files, computes
the similarity matrix, constructs a graph, applies PageRank, and selects the top sentences.

The results are stored in a dictionary where each key is the name of a parent directory, and each value
is a list containing information about each file, including the file name and the top-ranked sentences.

Usage:
- Adjust the 'parent_directory_path' variable to specify the path of the parent directory containing text files.
- Optionally modify 'num_sentences' to control the number of sentences to include in the summary.

Dependencies:
- sys
- os
- Scikit-learn
- NetworkX
"""



parent_directory_path = sys.argv[1]    # ../data/training
num_sentences = int(sys.argv[2])   # 5
output = sys.argv[3] # TextRankOutput



with open(output, "w") as o:
    # Dictionary to store content based on parent directory
    content_dict = {}

    for root, dirs, files in os.walk(parent_directory_path):

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
                            # ensure each sentence is unique, otherwise TextRank will output repeat sentences
                            if sentence not in sentlist:
                                sentlist.append(sentence)
                        else:
                            pass

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
                parent_directory = os.path.basename(root)
                if parent_directory not in content_dict.keys():
                    content_dict[parent_directory] = []

                """
                this dictionary format can be modified if it helps other stages. Currently, output form is:
                [{'file_name': 'AFP_ENG_20050119.0019', 'file_top_sent': ['Indian and Pakistani military commanders were to discuss Wednesday....', '...', ...]
                """    
                content_dict[parent_directory].append({
                    'file_name': file_name,
                    'file_top_sent': top_sentences
                })


    dir_list = content_dict.keys()
    for dir in dir_list:
        o.write(str(content_dict[dir]) + "\n")           



