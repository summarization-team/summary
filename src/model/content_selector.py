import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer


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
        """
        Retrieves the sentence embeddings for a list of sentences using the provided tokenizer and model.

        Args:
            sentlist (list): A list of sentences (str) to be embedded.
            tokenizer: A tokenizer object capable of tokenizing the sentences.
            model: A pre-trained model capable of generating sentence embeddings.

        Returns:
            list: A list of sentence embeddings generated by the model.
        """

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model.to(device)
        model.eval()  # Set the model to evaluation mode
        embeddings = []
        with torch.no_grad():
            tokenized_batch = tokenizer(sentlist, padding=True, truncation=True, return_tensors='pt')
            input_ids = tokenized_batch['input_ids'].to(device)
            attention_mask = tokenized_batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            
            # Extract the embeddings of the [CLS] token (first token of each sentence)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings.extend(cls_embeddings.cpu().numpy())
        return embeddings


    def select_content_textrank(self, all_documents, min_sent_len):
        """
        Selects top sentences from each document using TextRank algorithm based on sentence embeddings.

        Args:
            all_documents (dict): A dictionary containing document IDs as keys and their corresponding paragraphs and sentences as values.
            min_sent_len (int): Minimum length threshold for sentences to be considered.

        Returns:
            dict: A dictionary where each key is a document ID and the corresponding value is a list of top sentences selected by TextRank algorithm.
        """
        # Load pre-trained model tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Load pre-trained model
        model = BertModel.from_pretrained('bert-base-uncased')

        selected_sentences = {}

        for doc in all_documents.keys():
            sentlist = []
            sentlist_tok = []
            for paragraph in all_documents[doc].keys():
                if paragraph[0] == "P":
                    for sentence in all_documents[doc][paragraph]:
                        sentlist_tok.append(sentence)
                        sentence_str = " ".join(sentence)
                        sentlist.append(sentence_str)
            
            sent_toks_clean = [sentence for sentence in sentlist_tok if len(sentence)>= min_sent_len]
            sents_clean = [" ".join(sentence) for sentence in sent_toks_clean]
            embeddings = self.get_sentence_embeddings(sents_clean, tokenizer, model)
            # print("one document embeddings done")

            embeddings = np.array(embeddings)
            similarity_matrix = cosine_similarity(embeddings)

            # graph the resulting similarity matrix, then use the TextRank algorithm (thru PageRank) to find top sentence scores
            graph = nx.from_numpy_array(similarity_matrix)
            sentence_scores = nx.pagerank(graph, alpha = 0.85, max_iter = 100)
            ranked_sentindices = sorted(range(len(sentence_scores)), key=lambda index: sentence_scores[index], reverse=True)
            top_sentindices = ranked_sentindices[:self.num_sentences_per_doc]
            top_sentences = [sent_toks_clean[i] for i in top_sentindices]

            
            # stores top sentences as value in dictionary associated with the doc name as its key
            selected_sentences[doc] = top_sentences

        # compiled dictionary of the top n sentences for each document
        return selected_sentences
    
    def select_content_topic_focused(self, all_documents):
        """
        Selects top sentences from each document based on topic-focused approach.

        Args:
            all_documents (dict): A dictionary containing document IDs as keys and their corresponding paragraphs and sentences as values.

        Returns:
            dict: A dictionary where each key is a document ID and the corresponding value is a list of top sentences selected by the topic-focused approach.
        """
        # Load the pre-trained model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        selected_sentences = {}
        for doc in all_documents.keys():
            # Retrieve the document's headline
            headline = all_documents[doc][HEADLINE]
            # Retrieve the document's paragraphs and sentences
            paragraphs = [all_documents[doc][para] for para in all_documents[doc] if para != HEADLINE]
            # Flatten the list of sentences
            sentences = [sentence for paragraph in paragraphs for sentence in paragraph]
            # Compute the sentence embeddings
            sentence_embeddings = self.get_sentence_embeddings(sentences, self.tokenizer, self.model)
            headline_embedding = self.get_sentence_embeddings([headline], self.tokenizer, self.model)[0]
            # Compute the cosine similarity between the headline and each sentence
            similarities = cosine_similarity(sentence_embeddings, [headline_embedding])
            # Select the top n sentences based on their similarity to the headline
            top_indices = np.argsort(similarities.flatten())[-self.num_sentences_per_doc:]
            top_sentences = [sentences[i] for i in top_indices]
            # Store the selected sentences in the dictionary
            selected_sentences[doc] = top_sentences
        return selected_sentences
    
    def _preprocess_and_embed_with_titles(self, model, all_documents):
        all_embeddings = {}
        for doc in all_documents.keys():
            sentences = []
            if HEADLINE in all_documents[doc]:
                headline = " ".join(all_documents[doc][HEADLINE])  # Convert headline tokens to a string
                sentences.append(headline)
            for para in all_documents[doc]:
                if para == HEADLINE:
                    continue  # Skip since we've already processed the headline
                for sentence in all_documents[doc][para]:
                    flat_sentence = " ".join(sentence)
                    sentences.append(flat_sentence)
            # Encode all sentences, including the title
            embeddings = model.encode(sentences)
            all_embeddings[doc] = {'sentences': sentences, 'embeddings': embeddings}
        return all_embeddings

    def select_content(self, all_documents):
        """Selects content based on the specified approach."""
        if self.approach == 'tfidf':
            return self.select_content_tfidf(all_documents)
        elif self.approach == 'textrank':
            min_sent_len = 8
            return self.select_content_textrank(all_documents, min_sent_len)
        elif self.approach == 'topic_focused':
            return self.select_content_topic_focused(all_documents)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")
