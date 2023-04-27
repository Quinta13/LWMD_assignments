"""
TODO
"""
from __future__ import annotations

import heapq
from collections import Counter
from os import path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from io_ import get_files, read_jsonl, make_dir, save_sparse_matrix, save_dense_matrix, \
    load_sparse_matrix, load_dense_matrix, get_sparse_vector_files, get_vector_dir, get_dense_vector_files, \
    get_scores_dir, get_scores_files, get_images_dir, get_evaluation_dir, get_evaluation_file, save_dataframe, \
    get_evaluation_files
from settings import DEFAULT_LANGUAGE, CORPUS, QUERIES, EVALUATION_DIR, LOG
from utils import tokenize, top_k


class Document:
    """
    This class represent a document with its own doc-id and text
    """

    def __init__(self, id_: str, content: str, language: str = DEFAULT_LANGUAGE):
        """

        :param id_: doc-id
        :param content: document content
        :param language: language of the document (default in settings)
        """

        self._id: str = id_
        self._content: str = content
        self._language: str = language
        self._tokens: list[str] = []
        self._is_tokenized: bool = False

    def __str__(self) -> str:
        """
        :return: string representation of the object
        """
        return f"[{self._id}]"

    def __repr__(self) -> str:
        """
        :return: string representation of the object
        """
        return str(self)

    @property
    def id_(self) -> str:
        """
        :return: doc-id
        """
        return self._id

    @property
    def content(self) -> str:
        """
        :return: document content
        """
        return self._content

    @property
    def tokens(self) -> List[str]:
        """
        Return tokens of the document,
            the first time the method is invoked tokenization is applied
        :return: tokens
        """
        self._check_tokenized()
        return self._tokens

    @property
    def tokenized_content(self) -> str:
        """
        Return tokenized content
        :return: tokenized content
        """
        self._check_tokenized()
        return " ".join(self._tokens)

    @property
    def tf(self) -> Dict[str, int]:
        """
        Compute term frequency for the document
        :return: document term frequency
        """
        return dict(Counter(self._tokens))

    def tokenize(self):
        """
        Tokenize the document
        :return: tokenized document
        """
        if not self._is_tokenized:
            self._tokens = tokenize(text=self.content, language=self._language)
            self._is_tokenized = True

    def _check_tokenized(self):
        """
        Check if document was tokenized
        """
        if not self._is_tokenized:
            raise Exception("Document not tokenized yet")


class DocumentVectorizer:
    """
    This class produce both sparse and dense representation for given dataset
    """
    _ID_FIELD = '_id'
    _CONTENT_FIELD = 'text'

    def __init__(self, data_name: str):
        """

        :param data_name: dataset name in datasets folder
        """

        self._data_name = data_name

        self._documents: List[Document] = []
        self._queries: List[Document] = []

        self._docs_full_tokens: List[List[str]] = []
        self._queries_full_tokens: List[List[str]] = []

        self._tfidf_vectorizer: TfidfVectorizer = TfidfVectorizer()
        self._sparse_document_vectors: csr_matrix = csr_matrix([])
        self._sparse_query_vectors: csr_matrix = csr_matrix([])

        self._transformer: SentenceTransformer = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self._dense_document_vectors: np.ndarray = np.array([])
        self._dense_query_vectors: np.ndarray = np.array([])

        self._parsed: bool = False
        self._tokenized: bool = False
        self._sparse_vectorized: bool = False
        self._dense_vectorized: bool = False

    def parse(self):
        """
        Parse both documents and queries
        """
        self._parse_documents()
        self._parse_queries()
        self._parsed = True

    def _parse_documents(self):
        """
        Parse documents file to list of Document object
        """

        print("Parsing documents")

        file_, _, _ = get_files(self._data_name)

        try:
            docs_df: pd.DataFrame = read_jsonl(file_)
        except FileNotFoundError:
            raise Exception(f"Dataset folder is expected to contain {CORPUS}")

        self._documents: List[Document] = [
            Document(id_=row[self._ID_FIELD], content=row[self._CONTENT_FIELD])
            for _, row in docs_df.iterrows()
        ]

    def _parse_queries(self):
        """
        Parse queries file to list of Document object,
        """

        print("Parsing queries")

        _, file_, _ = get_files(self._data_name)

        try:
            queries_df: pd.DataFrame = read_jsonl(file_)
        except FileNotFoundError:
            raise Exception(f"Dataset folder is expected to contain {QUERIES}")

        self._queries: List[Document] = [
            Document(id_=row[self._ID_FIELD], content=row[self._CONTENT_FIELD])
            for _, row in queries_df.iterrows()
        ]

    @property
    def documents(self) -> List[Document]:
        """
        :return: List of parsed documents
        """
        self._check_parsed()
        return self._documents

    @property
    def queries(self) -> List[Document]:
        """
        :return: List of parsed documents
        """
        self._check_parsed()
        return self._queries

    def tokenize(self):
        """
        Tokenize documents, set on proper flag
        """

        print("Tokenizing documents")
        [doc.tokenize() for doc in self.documents]
        self._docs_full_tokens = [doc.tokens for doc in self.documents]

        print("Tokenizing queries")
        [q.tokenize() for q in self.queries]
        self._queries_full_tokens = [q.tokens for q in self.queries]

        self._tokenized = True

    def sparse_vectorize(self):
        """
        Create sparse representation for documents and queries
        """

        docs_corpus: List[str] = [doc.tokenized_content for doc in self.documents]
        query_corpus: List[str] = [q.tokenized_content for q in self.queries]

        print("Learning vocabulary idf ")

        self._tfidf_vectorizer.fit(raw_documents=docs_corpus)

        print("Generating document sparse vector ")

        self._sparse_document_vectors = self._tfidf_vectorizer.transform(docs_corpus)

        print("Generating query sparse vector ")

        self._sparse_query_vectors = self._tfidf_vectorizer.transform(query_corpus)

        self._sparse_vectorized = True

    @property
    def sparse_document_vectors(self) -> csr_matrix:
        """
        :return: sparse document vectors
        """
        self._check_sparse_vectorized()
        return self._sparse_document_vectors

    @property
    def sparse_query_vectors(self) -> csr_matrix:
        """
        :return: sparse query vectors
        """
        self._check_sparse_vectorized()
        return self._sparse_query_vectors

    def dense_vectorize(self):
        """
        Create dense representation for documents and queries
        """

        docs_corpus: List[str] = [doc.content for doc in self.documents]
        query_corpus: List[str] = [q.content for q in self.queries]

        print("Generating document dense vector ")

        self._dense_document_vectors = self._transformer.encode(docs_corpus)

        print("Generating query dense vector ")

        self._dense_query_vectors = self._transformer.encode(query_corpus)

        self._dense_vectorized = True

    @property
    def dense_document_vectors(self) -> np.ndarray:
        """
        :return: dense document vectors
        """
        self._check_dense_vectorized()
        return self._dense_document_vectors

    @property
    def dense_query_vectors(self) -> np.ndarray:
        """
        :return: dense query vectors
        """
        self._check_dense_vectorized()
        return self._dense_query_vectors

    def _check_parsed(self):
        """
        Check if documents were parsed
        """
        if not self._parsed:
            raise Exception("Documents not parsed yet")

    def _check_tokenized(self):
        """
        Check if documents were tokenized
        """
        if not self._tokenized:
            raise Exception("Documents not tokenized yet")

    def _check_sparse_vectorized(self):
        """
        Check if sparse vectorization was computed
        """
        if not self._sparse_vectorized:
            raise Exception("Not sparse vectorized yet")

    def _check_dense_vectorized(self):
        """
        Check if dense vectorization was computed
        """
        if not self._dense_vectorized:
            raise Exception("Not dense vectorized yet")

    def save(self):
        """
        Save sparse and dense representation to proper directory
        :return:
        """
        make_dir(get_vector_dir(self._data_name))
        self._save_sparse()
        self._save_dense()

    def _save_sparse(self):
        """
        Save sparse representation to proper directory
        """

        make_dir(get_vector_dir(self._data_name))

        out_doc, out_query = get_sparse_vector_files(data_name=self._data_name)

        save_sparse_matrix(mat=self.sparse_document_vectors, path_=out_doc)
        save_sparse_matrix(mat=self.sparse_query_vectors, path_=out_query)

    def _save_dense(self):
        """
        Save dense representation to proper directory
        """

        make_dir(get_vector_dir(self._data_name))

        out_doc, out_query = get_dense_vector_files(data_name=self._data_name)

        save_dense_matrix(mat=self.dense_document_vectors, path_=out_doc)
        save_dense_matrix(mat=self.dense_query_vectors, path_=out_query)


class ScoresComputation:
    """ This class use sparse and dense vectors to compute scores """

    def __init__(self, data_name: str):
        """

        :param data_name: dataset name in datasets folder
        """

        self._data_name = data_name

        self._sparse_docs: csr_matrix
        self._sparse_query: csr_matrix
        self._sparse_docs, self._sparse_query = self._load_sparse()

        self._dense_docs: np.ndarray
        self._dense_query: np.ndarray
        self._dense_docs, self._dense_query = self._load_dense()

        self._sparse_scores: np.ndarray = self._compute_sparse_score()
        self._dense_scores: np.ndarray = self._compute_dense_score()

    def _load_sparse(self) -> Tuple[csr_matrix, csr_matrix]:
        """
        :return: sparse representation for documents and queries
        """

        in_docs, in_queries = get_sparse_vector_files(data_name=self._data_name)

        try:
            return load_sparse_matrix(path_=in_docs), \
                   load_sparse_matrix(path_=in_queries)
        except FileNotFoundError:
            raise Exception("No sparse vector found, use DocumentVectorizer to generate")

    def _load_dense(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: dense representation for documents and queries
        """

        in_docs, in_queries = get_dense_vector_files(data_name=self._data_name)

        try:
            return load_dense_matrix(path_=in_docs), \
                   load_dense_matrix(path_=in_queries)
        except FileNotFoundError:
            raise Exception("No dense vector found, use DocumentVectorizer to generate")

    @property
    def sparse_docs(self) -> csr_matrix:
        """
        :return: sparse document vectors
        """
        return self._sparse_docs

    @property
    def sparse_query(self) -> csr_matrix:
        """
        :return: sparse query vectors
        """
        return self._sparse_query

    @property
    def dense_docs(self) -> np.ndarray:
        """
        :return: dense document vectors
        """
        return self._dense_docs

    @property
    def dense_query(self) -> np.ndarray:
        """
        :return: dense query vectors
        """
        return self._dense_query

    def _compute_sparse_score(self) -> np.ndarray:
        """
        Compute sparse scores
        :return: sparse scores vectors
        """
        return self.sparse_query.dot(self.sparse_docs.transpose()).toarray()
        # return cosine_similarity(X=self.sparse_query, Y=self.sparse_docs)

    def _compute_dense_score(self) -> np.ndarray:
        """
        Compute dense scores
        :return: dense scores vectors
        """
        return self.dense_query.dot(self.dense_docs.transpose())
        # return cosine_similarity(X=self.dense_query, Y=self.dense_docs)

    @property
    def sparse_scores(self) -> np.ndarray:
        """
        :return: sparse score matrix
        """
        return self._sparse_scores

    @property
    def dense_scores(self) -> np.ndarray:
        """
        :return: sparse score matrix
        """
        return self._dense_scores

    @property
    def full_scores(self) -> np.ndarray:
        return self.sparse_scores + self.dense_scores

    def save(self):
        """
        Save score matrix locally to proper directory
        """

        make_dir(get_scores_dir(data_name=self._data_name))

        out_sparse, out_dense, out_full = get_scores_files(self._data_name)

        save_dense_matrix(mat=self.sparse_scores, path_=out_sparse)
        save_dense_matrix(mat=self.dense_scores, path_=out_dense)
        save_dense_matrix(mat=self.full_scores, path_=out_full)


class RecallEvaluation:
    """ This class compute the recall combining varying top-k' scores for dense and sparse vectors """

    _PLOT_NAME = "execution"
    _PLOT_COLOR = "blue"
    _PLOT_LINES = "orange"
    _PLOT_THRESHOLD = .9

    def __init__(self, data_name: str, k: int):
        """

        :param data_name: dataset name in datasets folder
        :param k: number of elements in tok-k ground through
        """

        self._data_name: str = data_name
        self._k = k

        self._sparse_scores: np.ndarray
        self._dense_scores: np.ndarray
        self._full_scores: np.ndarray
        self._sparse_scores, self._dense_scores, self._full_scores = self._load_scores()

        self._ground_truth = self._get_ground_truth()

        self._results: pd.DataFrame = pd.DataFrame([])
        self._evaluated: bool = False

    def _load_scores(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load scores files
        :return: sparse scores, dense scores, full scores
        """

        in_sparse, in_dense, in_full = get_scores_files(self._data_name)

        try:
            return load_dense_matrix(path_=in_sparse), \
                   load_dense_matrix(path_=in_dense), \
                   load_dense_matrix(path_=in_full)
        except FileNotFoundError:
            raise Exception("No score files found, use ScoresComputation to generate")

    def _get_ground_truth(self) -> np.ndarray:
        """
        Return actual top-k documents as ground-truth for each query
        :return: actual top-k documents
        """

        return np.stack([top_k(scores, self._k) for scores in self._full_scores])

    @property
    def ground_truth(self) -> np.ndarray:
        """
        :return: ground truth (actual top-k document)
        """
        return self._ground_truth

    def _get_query_varying_recall(self, query_id: int) -> Dict[int, float]:
        """
        Evaluate recall varying k-prime until reaching top recall
        :param query_id: id of query (corresponding to the row)
        """

        k_prime_recall: Dict[int, float] = dict()

        # scores for the query
        full_scores = self._full_scores[query_id]

        # query ground truth
        ground_truth = self.ground_truth[query_id]

        # doc_ids sorted by decreasing score
        docs_sparse = np.argsort(self._sparse_scores[query_id])[::-1]
        docs_dense = np.argsort(self._dense_scores[query_id])[::-1]

        # extracted docs
        extracted = set()

        # actual top-k documents
        top_k = set()

        # max heap (score, in ground-truth, doc-id)
        maxheap: List[float, bool, int] = []
        heapq.heapify(maxheap)

        # starting values
        recall = 0
        k_prime = 0

        while recall != 1.:

            k_prime += 1

            # docs found with augmented k_prime
            extracted_docs = {docs_sparse[k_prime - 1], docs_dense[k_prime - 1]}

            # yet found docs not extracted yet
            new_docs = list(extracted_docs.difference(extracted))

            # adding new docs found to extracted ones
            extracted = extracted.union(extracted_docs)

            for doc in new_docs:

                # looking at exact value
                full_score_doc = (full_scores[doc], doc in ground_truth, doc)

                # if max-heap contains less then K elements keep adding
                if len(maxheap) < self._k:
                    heapq.heappush(maxheap, full_score_doc)
                    top_k.add(doc)
                # otherwise check if can enter max-heap substituting the minium
                else:
                    _, _, removed_doc = heapq.heappushpop(maxheap, full_score_doc)
                    # if heap has been modified, so top-k does
                    if removed_doc != doc:
                        top_k.remove(removed_doc)
                        top_k.add(doc)

            # computing recall
            relevant_in_top_k = top_k.intersection(ground_truth)
            recall = len(relevant_in_top_k) / self._k

            k_prime_recall[k_prime] = recall

        return k_prime_recall

    def evaluate(self):
        """
        Evaluate recall at varying k-prime for each query
        """

        n_queries = len(self._full_scores)

        results = []
        highest_k = 0  # saving highest k-prime found

        for query_id in range(n_queries):
            results.append(self._get_query_varying_recall(query_id=query_id))

            highest_k = max(highest_k, list(results[-1].keys())[-1])

        #  aligning all results to same k (the highest found)
        new_results = [
            {k: result.get(k, 1) for k in range(1, highest_k + 1)}
            for result in results
        ]

        self._evaluated = True

        self._results = pd.DataFrame(new_results)

    @property
    def results(self) -> pd.DataFrame:
        """
        :return: results
        """
        self._check_evaluated()
        return self._results

    def plot(self, save: bool = False):
        """
        Plot k-prime trend
        :param save: if to save
        :param file_name: name of file
        """

        self._check_evaluated()

        results_dict = self.results.mean(axis=0).to_dict()

        x = list(results_dict.keys())
        y = list(results_dict.values())

        first_one = next(x[0] for x in enumerate(y) if x[1] > self._PLOT_THRESHOLD) + 1
        y_first_one = y[first_one - 1]

        plt.plot(x, y, color=self._PLOT_COLOR)

        # Set the axis labels and chart title
        plt.title(f'Top-{self._k} retrieval')
        plt.xlabel('k\'')
        plt.ylabel('Recall')

        # Red line
        plt.plot([0, first_one], [y_first_one, y_first_one], color='orange')
        plt.plot([first_one, first_one], [0, y_first_one], color='orange')

        plt.xticks(list(plt.xticks()[0][1:]) + [first_one])
        plt.gca().get_xticklabels()[-1].set_color(self._PLOT_LINES)
        plt.xticks(rotation=-90)

        plt.yticks(list(plt.yticks()[0][1:]) + [round(y_first_one, 3)])
        plt.gca().get_yticklabels()[-1].set_color(self._PLOT_LINES)

        plt.plot(
            results_dict.keys(),
            results_dict.values()
        )

        if save:
            images_dir = get_images_dir(data_name=self._data_name)
            make_dir(path_=images_dir)

            out_file_name = f"{self._PLOT_NAME}_k{self._k}.svg"
            out_file = path.join(images_dir, out_file_name)

            if LOG:
                print(f"Saving image {out_file}")
            plt.savefig(out_file, format='svg', dpi=1200)

        plt.show()

    def save(self):
        """
        Saves result in proper directory
        """

        make_dir(path_=get_evaluation_dir(self._data_name))

        evaluation_file = get_evaluation_file(data_name=self._data_name, k=self._k)

        save_dataframe(df=self.results, path_=evaluation_file)

    def _check_evaluated(self):
        if not self._evaluated:
            raise Exception("Model not evaluated")


class RecallAnalysis:
    """ This class provides some methods to analyze interaction within recall and k'"""

    _THRESHOLDS = [i * .1 for i in range(1, 10+1)]
    _PLOT_NAME = "kprime_recall"

    def __init__(self, data_name: str):
        """
        :param data_name: dataset name in datasets folder
        """

        self._data_name: str = data_name

        self._indexes: List[Tuple[int, List[int]]] = self._get_indexes()

    def _get_indexes(self) -> List[Tuple[int, List[int]]]:
        """
        :return: list of k associated to list of indexes corresponding to certain threshold
        """

        # reading evaluation files
        try:
            evaluations: List[Tuple[int, pd.DataFrame]] = get_evaluation_files(data_name=self._data_name)
        except FileNotFoundError:
            raise Exception(f"No directory {EVALUATION_DIR}. Use RecallEvaluation to create")

        indexes = [(x, self._get_threshold_indexes(y)) for x, y in evaluations]

        return indexes

    def _get_threshold_indexes(self, df: pd.DataFrame) -> List[int]:
        """
        Given a results dataframe returns indexes corresponding to each recall threshold
        :param df: results dataframe
        :return: list of indexes corresponding to recall thresholds
        """
        arr = df.mean(axis=0)
        indexes = [
            next(x[0] for x in enumerate(arr) if x[1] >= i) + 1
            for i in self._THRESHOLDS
        ]
        return indexes

    def plot(self, save=False):
        """
        Plot interaction between k' and recall
        :param save: if to save plot in images folder
        """

        for k, y in self._indexes:
            plt.plot(self._THRESHOLDS, y, label=k)

        plt.title("Interaction between k' and recall")
        plt.xlabel("Recall")
        plt.ylabel("k-prime")
        leg = plt.legend(title="k")
        leg._legend_box.align = "right"

        if save:
            images_dir = get_images_dir(data_name=self._data_name)
            make_dir(path_=images_dir)

            out_file_name = f"{self._PLOT_NAME}.svg"
            out_file = path.join(images_dir, out_file_name)

            if LOG:
                print(f"Saving image {out_file}")
            plt.savefig(out_file, format='svg', dpi=1200)

        plt.show()
