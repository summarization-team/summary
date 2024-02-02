import math
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np


HEADLINE = 'HEADLINE'
DATELINE = 'DATELINE'
PARAGRAPH = 'PARAGRAPH'


class ContentSelector:
    def __init__(self, num_sentences_per_doc, approach='tfidf'):
        self.num_sentences_per_doc = num_sentences_per_doc
        self.approach = approach

    def _flatten_sentences_with_headlines(self, documents):
        """
        Flatten all sentences across all documents, 
        maintaining their document and paragraph context.
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
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        return tfidf_matrix
    
    def _select_top_sentences(self, tfidf_matrix, doc_sent_mapping):
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
        # Preprocess the input including headlines
        corpus, doc_sent_mapping = self._flatten_sentences_with_headlines(all_documents)

        # Compute TF-IDF including headlines
        tfidf_matrix = self._compute_tfidf(corpus)

        # Select top n sentences per document including consideration for headlines
        top_sentences = self._select_top_sentences(tfidf_matrix, doc_sent_mapping)

        # Output
        return top_sentences
        

    def select_content_textrank(self, all_documents):
        """
        Performs TextRank algorithm on each document, returning top 'num_sentences_per_doc' sentences
        for each document and compiling them into selected_sentences, a list of the top n sentences of each document.
        """
        selected_sentences = []

        for doc in all_documents:

            sentlist = [sentence for para in doc for sentence in para]

            # create vectors for each sentence, use cosine similarity to compare them
            vectorizer = CountVectorizer(stop_words="english")
            sentence_vectors = vectorizer.fit_transform(sentlist)
            similarity_matrix = cosine_similarity(sentence_vectors)

            # graph the resulting similarity matrix, then use the TextRank algorithm (thru PageRank) to find top sentence scores
            graph = nx.from_numpy_array(similarity_matrix)
            sentence_scores = nx.pagerank(graph, alpha = 0.85, max_iter = 100) 
            
            ranked_sentindices = sorted(range(len(sentence_scores)), key=lambda index: sentence_scores[index], reverse=True)
            top_sentindices = ranked_sentindices[:self.num_sentences_per_doc]

            top_sentences = [sentlist[i] for i in top_sentindices]

            selected_sentences.extend(top_sentences)

        # compiled list of sentences containing the top n sentences per document
        return selected_sentences

    def select_content(self, all_documents):
        """Selects content based on the specified approach."""
        if self.approach == 'tfidf':
            return self.select_content_tfidf(all_documents)
        elif self.approach == 'textrank':
            return self.select_content_textrank(all_documents)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")

