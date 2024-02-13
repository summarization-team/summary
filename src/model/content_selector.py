import math
import networkx as nx
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

HEADLINE = 'HEADLINE'
PARAGRAPH = 'PARAGRAPH'


class ContentSelector:
    """
    A class to select the most important content from a collection of documents.
    """
    def __init__(self, num_sentences_per_doc, approach='tfidf'):
        """
        Initialize the content selector.
        """
        self.num_sentences_per_doc = num_sentences_per_doc
        self.approach = approach

    def _flatten_sentences_with_headlines(self, documents):
        """
        Flatten all sentences across all documents, 
        maintaining their document and paragraph context.
        Args:
        - documents (dict): A dictionary containing document information. 
                            Each document is identified by a key, and its value is another dictionary.
                            The inner dictionary has paragraph keys, and each paragraph contains a list of sentences.
        Returns:
        - list: A list of all sentences in the documents, including headlines.
        - list: A list of tuples, each containing the document name, sentence, and paragraph name.
        """
        corpus = []
        doc_sent_mapping = []
        for doc in documents:
            if HEADLINE in documents[doc]:
                headline = " ".join(documents[doc][HEADLINE])  # Convert headline tokens to a string
                corpus.append(headline)  # Add headline to the corpus
                doc_sent_mapping.append((doc, documents[doc][HEADLINE], "headline"))
            for para in documents[doc]:
                if para == HEADLINE:
                    continue  # Skip since we've already processed the headline
                for sentence in documents[doc][para]:
                    flat_sentence = " ".join(sentence)
                    corpus.append(flat_sentence)
                    doc_sent_mapping.append((doc, sentence, para))
        return corpus, doc_sent_mapping
    
    def _compute_tfidf(self,corpus):
        """
        Compute TF-IDF scores for a given corpus.
        Args:
        - corpus (list): A list of sentences.
        Returns:
        - matrix: A matrix of TF-IDF scores for each sentence in the corpus.
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        return tfidf_matrix
    
    def _select_top_sentences(self, tfidf_matrix, doc_sent_mapping):
        """
        Select the top sentences from a collection of documents based on their TF-IDF scores.
        Args:
        - tfidf_matrix: A matrix of TF-IDF scores for each sentence in the corpus.
        - doc_sent_mapping: A list of tuples, each containing the document name, sentence, and paragraph name.
        Returns:
        - dict: A dictionary containing the top-ranked sentences for each document.
            The keys are document names, and the values are lists of selected sentences.
        """
        doc_scores = {}
        # Initialize doc_scores dictionary to store individual sentence scores per document
        for idx, (doc, sentence, _) in enumerate(doc_sent_mapping):
            if doc not in doc_scores:
                doc_scores[doc] = []
            sentence_score = np.sum(tfidf_matrix[idx].toarray())
            doc_scores[doc].append((sentence, sentence_score))

        top_sentences = {}
        # For each document, sort its sentences by their scores and select the top n
        for doc, scores in doc_scores.items():
            sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
            top_sentences[doc] = [sentence for sentence, _ in sorted_scores[:self.num_sentences_per_doc]]
        
        return top_sentences

    def select_content_tfidf(self, all_documents):
        """
        Apply TF-IDF algorithm to extract the top sentences from a collection of documents.
        Args:
        - all_documents (dict): A dictionary containing document information. 
                            Each document is identified by a key, and its value is another dictionary.
                            The inner dictionary has paragraph keys, and each paragraph contains a list of sentences.
        Returns:
        - dict: A dictionary containing the top-ranked sentences for each document.
                The keys are document names, and the values are lists of selected sentences.
        """
        # Preprocess the input including headlines
        corpus, doc_sent_mapping = self._flatten_sentences_with_headlines(all_documents)

        # Compute TF-IDF including headlines
        tfidf_matrix = self._compute_tfidf(corpus)

        # Select top n sentences per document including consideration for headlines
        top_sentences = self._select_top_sentences(tfidf_matrix, doc_sent_mapping)

        # Output
        return top_sentences
        




    def get_sentence_embeddings(self, sentlist, tokenizer, model):
        embeddings = []
        for sentence in sentlist:
            tokens = tokenizer.tokenize(sentence)
            # add a classification token to help obtain fixed-size representation of sentence
            # add a separator token to separate sentences
            tokens = ['[CLS]'] + tokens + ['[SEP]']
            id_vector = tokenizer.convert_tokens_to_ids(tokens)
            # truncate or pad to make sure vector fits within BERT's sequence length limit
            id_vector = id_vector[:512] + [0] * (512 - len(id_vector))

            # get embedding from BERT model
            with torch.no_grad():
                outputvecs = model(torch.tensor([id_vector]))[0]
            
            # average embeddings for the tokens, exclude the CLS and SEP special tokens on either end
            sent_embed = torch.mean(outputvecs[:,1:-1], dim=1).squeeze().numpy()
            embeddings.append(sent_embed)
        
        return embeddings
    


    def select_content_textrank(self, all_documents):

        # Load pre-trained model tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Load pre-trained model
        model = BertModel.from_pretrained('bert-base-uncased')
        model.config
        model.eval()  # Set the model to evaluation mode

        selected_sentences = {}

        for doc in all_documents.keys():
            sentlist = []
            sentlist_tok = []
            for paragraph in all_documents[doc].keys():
                for sentence in all_documents[doc][paragraph]:
                    sentlist_tok.append(sentence)
                    sentence_str = " ".join(sentence)
                    sentlist.append(sentence_str)
            
            embeddings = self.get_sentence_embeddings(sentlist, tokenizer, model)

            embeddings = np.array(embeddings)
            similarity_matrix = cosine_similarity(embeddings)

            # graph the resulting similarity matrix, then use the TextRank algorithm (thru PageRank) to find top sentence scores
            graph = nx.from_numpy_array(similarity_matrix)
            sentence_scores = nx.pagerank(graph, alpha = 0.85, max_iter = 100) 
            
            ranked_sentindices = sorted(range(len(sentence_scores)), key=lambda index: sentence_scores[index], reverse=True)
            top_sentindices = ranked_sentindices[:self.num_sentences_per_doc]

            top_sentences = [sentlist_tok[i] for i in top_sentindices]


            # stores top sentences as value in dictionary associated with the doc name as its key
            selected_sentences[doc] = top_sentences

        # compiled dictionary of the top n sentences for each document
        return selected_sentences



    # def select_content_textrank(self, all_documents):
    #     """
    #     Apply TextRank algorithm to extract the top sentences from a collection of documents.

    #     Parameters:
    #     - all_documents (dict): A dictionary containing document information. 
    #                         Each document is identified by a key, and its value is another dictionary.
    #                         The inner dictionary has paragraph keys, and each paragraph contains a list of sentences.

    #     Returns:
    #     dict: A dictionary containing the top-ranked sentences for each document.
    #         The keys are document names, and the values are lists of selected sentences.

    #     Notes:
    #     This function uses the TextRank algorithm to rank sentences within each document based on cosine similarity.
    #     It constructs a similarity matrix between sentences, applies the PageRank algorithm,
    #     and selects the top sentences as representatives of the document's content.

    #     The number of top sentences to be extracted per document is determined by the `num_sentences_per_doc` attribute.

    #     Example:
    #     ```
    #     all_documents = {
    #         'doc1': {
    #             'paragraph1': ['sentence1', 'sentence2', ...],
    #             'paragraph2': ['sentence3', 'sentence4', ...],
    #             ...
    #         },
    #         'doc2': {
    #             'paragraph1': ['sentence5', 'sentence6', ...],
    #             'paragraph2': ['sentence7', 'sentence8', ...],
    #             ...
    #         },
    #         ...
    #     }
    #     """

    #     selected_sentences = {}

    #     for doc in all_documents.keys():
    #         sentlist = []
    #         sentlist_tok = []
    #         for paragraph in all_documents[doc].keys():
    #             for sentence in all_documents[doc][paragraph]:
    #                 sentlist_tok.append(sentence)
    #                 sentence_str = " ".join(sentence)
    #                 sentlist.append(sentence_str)

    #         # create vectors for each sentence, use cosine similarity to compare them
    #         vectorizer = CountVectorizer(stop_words="english")
    #         sentence_vectors = vectorizer.fit_transform(sentlist)
    #         similarity_matrix = cosine_similarity(sentence_vectors)

    #         # graph the resulting similarity matrix, then use the TextRank algorithm (thru PageRank) to find top sentence scores
    #         graph = nx.from_numpy_array(similarity_matrix)
    #         sentence_scores = nx.pagerank(graph, alpha = 0.85, max_iter = 100) 
            
    #         ranked_sentindices = sorted(range(len(sentence_scores)), key=lambda index: sentence_scores[index], reverse=True)
    #         top_sentindices = ranked_sentindices[:self.num_sentences_per_doc]

    #         top_sentences = [sentlist_tok[i] for i in top_sentindices]


    #         # stores top sentences as value in dictionary associated with the doc name as its key
    #         selected_sentences[doc] = top_sentences

    #     # compiled dictionary of the top n sentences for each document
    #     return selected_sentences
    

    def select_content(self, all_documents):
        """Selects content based on the specified approach."""
        if self.approach == 'tfidf':
            return self.select_content_tfidf(all_documents)
        elif self.approach == 'textrank':
            return self.select_content_textrank(all_documents)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")

