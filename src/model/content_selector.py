import networkx as nx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize

HEADLINE = 'HEADLINE'
PARAGRAPH = 'PARAGRAPH'
DESCRIPTION = 'DESCRIPTION'
DATELINE = 'DATELINE'


class ContentSelector:
    """d
    A class to select the most important content from a collection of documents.
    """

    def __init__(self, num_sentences_per_doc, approach='tfidf', similarity_threshold=0.5, model_id=None):
        """
        Initialize the content selector.
        """
        self.num_sentences_per_doc = num_sentences_per_doc
        self.approach = approach
        self.similarity_threshold = similarity_threshold
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self.device = None

        if approach == 'textrank':
            self.set_model(model_id)
            self.set_tokenizer(model_id)
            self.set_device()
            self.model.to(self.device)
            self.model.eval()  # Set the model to evaluation mode
            self.output_device_name()
        elif approach == 'topic_focused':
            self.set_model(model_id)

    def set_model(self, model_id):
        if self.approach == 'textrank':
            self.model = AutoModel.from_pretrained(model_id)
        elif self.approach == 'topic_focused':
            self.model = SentenceTransformer(model_id)

    def set_tokenizer(self, model_id):
        # Load pre-trained model tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def set_device(self):
        self.device = torch.device(0 if torch.cuda.is_available() else "cpu")

    def output_device_name(self):
        with open('condor_logs/D5_gpu_selector.test', 'w', encoding='utf-8') as outfile:
            outfile.write(f"device={self.device}")

    def _flatten_sentences_with_headlines(self, documents):
        """
        Flatten all sentences across all documents, 
        maintaining their document and paragraph context.
        Args:
        - documents (dict): A dictionary containing document information. 
                            Each document is identified by a key, and its value is another dictionary.
                            The inner dictionary has paragraph keys, and each paragraph contains a list of sentences.
        Returns:c
        - list: A list of all sentences in the documents, including headlines.
        - list: A list of tuples, each containing the document name, sentence, and paragraph name.
        """
        corpus = []
        doc_sent_mapping = []
        for doc in documents:
            if DESCRIPTION in doc:
                continue  # Skip since we don't want to include description sentences in the summary
            if HEADLINE in documents[doc]:
                headline = documents[doc][HEADLINE]  # Convert headline tokens to a string
                corpus.append(headline)  # Add headline to the corpus
                doc_sent_mapping.append((doc, documents[doc][HEADLINE], "headline"))
            for para in documents[doc]:
                if para == HEADLINE or para == DATELINE:
                    continue  # Skip since we've already processed the headline
                for sentence in documents[doc][para]:
                    flat_sentence = " ".join(sentence)
                    corpus.append(flat_sentence)
                    doc_sent_mapping.append((doc, sentence, para))
        return corpus, doc_sent_mapping

    def _compute_tfidf(self, corpus):
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

    def get_sentence_embeddings(self, sentlist, tokenizer, model, device):
        """
        Retrieves the sentence embeddings for a list of sentences using the provided tokenizer and model.

        Args:
            sentlist (list): A list of sentences (str) to be embedded.
            tokenizer: A tokenizer object capable of tokenizing the sentences.
            model: A pre-trained model capable of generating sentence embeddings.

        Returns:
            list: A list of sentence embeddings generated by the model.
        """

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


        selected_sentences = {}

        for doc in all_documents.keys():
            if DESCRIPTION in doc or doc == "description.txt":
                continue
            sentlist = []
            sentlist_tok = []
            for paragraph in all_documents[doc].keys():
                if paragraph[0] == "P":
                    for sentence in all_documents[doc][paragraph]:
                        sentlist_tok.append(sentence)
                        sentence_str = " ".join(sentence)
                        sentlist.append(sentence_str)

            sent_toks_clean = [sentence for sentence in sentlist_tok if len(sentence) >= min_sent_len]
            sents_clean = [" ".join(sentence) for sentence in sent_toks_clean]
            embeddings = self.get_sentence_embeddings(sents_clean, self.tokenizer, self.model, self.device)
            # print("one document embeddings done")

            embeddings = np.array(embeddings)
            similarity_matrix = cosine_similarity(embeddings)

            # graph the resulting similarity matrix, then use the TextRank algorithm (thru PageRank) to find top sentence scores
            graph = nx.from_numpy_array(similarity_matrix)
            sentence_scores = nx.pagerank(graph, alpha=0.85, max_iter=100)
            ranked_sentindices = sorted(range(len(sentence_scores)), key=lambda index: sentence_scores[index],
                                        reverse=True)
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
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        model = SentenceTransformer('paraphrase-distilroberta-base-v1')

        # Preprocess and embed the documents
        all_embeddings = self._preprocess_and_embed_with_titles(model, all_documents)

        # Calculate the topic embedding
        topic_embedding = model.encode([all_documents[DESCRIPTION]])[0]

        # Calculate the similarities between the topic and the document sentences
        self._calculate_similarities(all_embeddings, topic_embedding)

        # Apply LexRank to select the top sentences based on the similarities
        return self._apply_lexrank(all_embeddings)

    def _preprocess_and_embed_with_titles(self, model, all_documents):
        """
        Preprocesses and embeds the documents using the provided model.
        Args:
        - model: A pre-trained model capable of generating sentence embeddings.
        - all_documents (dict): A dictionary containing document information.
        Returns:
        - dict: A dictionary containing the preprocessed and embedded documents.
        """
        all_embeddings = {}
        for doc in all_documents.keys():
            sentences = []
            if DESCRIPTION in doc:
                continue  # Skip since we don't want to include description sentences in the summary
            if HEADLINE in all_documents[doc]:
                sentences.append(all_documents[doc][HEADLINE])
            for para in all_documents[doc]:
                if para == HEADLINE or para == DATELINE:
                    continue  # Skip since we've already processed the headline
                for sentence in all_documents[doc][para]:
                    flat_sentence = " ".join(sentence)
                    sentences.append(flat_sentence)
            # Encode all sentences, including the title
            embeddings = model.encode(sentences)
            all_embeddings[doc] = {'sentences': sentences, 'embeddings': embeddings}
        return all_embeddings

    def _calculate_similarities(self, all_embeddings, topic_embedding):
        """
        Calculates the similarities between the topic and the document sentences.d
        Args:
        - all_embeddings (dict): A dictionary containing the preprocessed and embedded documents.
        - topic_embedding (np.array): The embedding of the topic.
        """
        for doc_id, data in all_embeddings.items():
            embeddings = data['embeddings']
            # Calculate similarity with the topic
            topic_similarities = cosine_similarity(embeddings, [topic_embedding]).flatten()
            # Calculate inter-sentence similarity
            sentence_similarities = cosine_similarity(embeddings)

            all_embeddings[doc_id]['topic_similarities'] = topic_similarities
            all_embeddings[doc_id]['sentence_similarities'] = sentence_similarities

    def _apply_lexrank(self, all_embeddings):
        """
        Applies LexRank to select the top sentences based on the similarities.
        Args:
        - all_embeddings (dict): A dictionary containing the preprocessed and embedded documents.
        Returns:
        - dict: A dictionary where each key is a document ID and the corresponding value is a list of top sentences selected by LexRank.
        """
        selected_sentences = {}
        for doc_id, data in all_embeddings.items():
            sentence_similarities = data['sentence_similarities']
            # Apply threshold
            graph = np.where(sentence_similarities >= self.similarity_threshold, 1, 0)

            # Calculate LexRank scores (simplified example using degree centrality)
            scores = np.sum(graph, axis=0)

            if np.sum(scores) == 0:  # Fallback if no sentences have scores
                ranked_indices = range(min(self.num_sentences_per_doc, len(data['sentences'])))
            else:
                ranked_indices = np.argsort(scores)[-self.num_sentences_per_doc:]

            sentences = [data['sentences'][i] for i in ranked_indices]
            top_sentences = []
            for sentence in sentences:
                if len(sentence) > 0:
                    top_sentences.append(word_tokenize(sentence))
            selected_sentences[doc_id] = top_sentences

        return selected_sentences

    def select_content_baseline(self, all_documents):
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
        # Sort the keys alphabetically
        sorted_doc_ids = sorted([x for x in all_documents.keys() if x[-1].isdigit()])

        # Pick the first key
        first_doc = sorted_doc_ids[0]

        selected_document = all_documents[first_doc]

        selected_sentences = {}
        selected_sentences[first_doc] = []

        for key, list_of_sentences in selected_document.items():
            if 'PARAGRAPH' in key:
                for sentence in list_of_sentences:
                    if len(selected_sentences[first_doc]) <= 5:
                        selected_sentences[first_doc].append(sentence)

        return selected_sentences

    def select_content(self, all_documents):
        """
        Selects top sentences from each document based on the specified approach.
        Args:
        - all_documents (dict): A dictionary containing document IDs as keys and their corresponding paragraphs and sentences as values.
        """
        if self.approach == 'tfidf':
            return self.select_content_tfidf(all_documents)
        elif self.approach == 'textrank':
            min_sent_len = 8
            return self.select_content_textrank(all_documents, min_sent_len)
        elif self.approach == 'topic_focused':
            return self.select_content_topic_focused(all_documents)
        elif self.approach == 'baseline':
            return self.select_content_baseline(all_documents)
        else:
            raise ValueError(f"Unknown approach: {self.approach}")
