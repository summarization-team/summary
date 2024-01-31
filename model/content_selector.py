import math
import networkx as nx
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

class ContentSelector:
    def __init__(self, num_sentences_per_doc, approach='tfidf'):
        self.num_sentences_per_doc = num_sentences_per_doc
        self.approach = approach
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def preprocess(self, text):
        """Preprocesses the text by tokenization, stemming, and removing stopwords."""
        words = word_tokenize(text)
        words = [self.stemmer.stem(w.lower()) for w in words if w.isalpha() and w.lower() not in self.stop_words]
        return words

    def compute_tf(self, document):
        """Computes Term Frequency (TF) for a given document."""
        tf_scores = {}
        word_counts = Counter(document)
        total_words = len(document)
        for word, count in word_counts.items():
            tf_scores[word] = count / float(total_words)
        return tf_scores

    def compute_idf(self, all_documents):
        """Computes Inverse Document Frequency (IDF) for all documents."""
        idf_scores = {}
        total_docs = sum(len(docs) for docs in all_documents)
        word_doc_counts = Counter(word for docs in all_documents for doc in docs for word in set(doc))
        for word, count in word_doc_counts.items():
            idf_scores[word] = math.log(total_docs / float(count))
        return idf_scores

    def select_content_tfidf(self, all_documents):
        """Selects the most relevant content from all documents based on TF-IDF scores."""
        flattened_docs = [sentence for doc in all_documents for para in doc for sentence in para]
        processed_docs = [self.preprocess(doc) for doc in flattened_docs]
        idf_scores = self.compute_idf(processed_docs)

        doc_sentences_scores = []
        for doc in all_documents:
            doc_scores = []
            for para in doc:
                for sentence in para:
                    processed_sentence = self.preprocess(sentence)
                    tf_scores = self.compute_tf(processed_sentence)
                    tf_idf_score = sum(tf_scores[word] * idf_scores.get(word, 0) for word in processed_sentence)
                    doc_scores.append((tf_idf_score, sentence))
            doc_sentences_scores.append(doc_scores)

        selected_sentences = []
        for doc_scores in doc_sentences_scores:
            doc_scores.sort(key=lambda x: x[0], reverse=True)
            top_sentences = [sentence for _, sentence in doc_scores[:self.num_sentences_per_doc]]
            selected_sentences.extend(top_sentences)

        return selected_sentences

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

# Example usage:
# selector = ContentSelector(num_sentences_per_doc=3, approach='textrank')
# selected_content = selector.select_content(all_documents)
