"""
This class provides some classes for sequential pairwise-similarity evaluation
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Dict

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from assignment3.io_ import log, load_vectors, make_dir, get_evaluation_dir, \
    get_exact_solution_file, save_evaluation, load_evaluation, check_exact_evaluation, get_evaluation_file, \
    get_vector_file
from assignment3.model.documents import DocumentVectors
from assignment3.settings import EXACT_SOLUTION
from assignment3.utils import jaccard


class SimilarityPairsEvaluation(ABC):
    """ This class compute exact solution """

    TIME_KEY = 'execution_time'
    # PREPROCESSING_KEY = 'preprocessing_time'
    THRESHOLD_KEY = 'threshold'
    PAIRS_KEY = 'pairs'

    _CLASS_NAME = "SimilarityPairsEvaluation"

    # DUNDER

    def __init__(self, data_name: str, threshold: float):
        """
        :param data_name: dataset name in datasets folder
        :param threshold: similarity threshold
        """

        self._data_name = data_name

        if not 0 <= threshold <= 1:
            raise Exception(f"Invalid threshold {threshold}: not in range [0, 1] ")

        self._threshold = threshold

        self._document_vectors: DocumentVectors = \
            DocumentVectors(data_name=self._data_name)  # delegate on download and vectorization check

        self._results: Dict = dict()  # available after evaluation
        self._evaluated: bool = False

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """
        return f"{self._CLASS_NAME} - {self._data_name} ({len(self)} docs, {self._threshold} similarity) "

    def __repr__(self):
        """
        :return: string representation for the object
        """
        return str(self)

    def __len__(self):
        """
        :return: len of documents to evaluate
        """
        return len(self._document_vectors)

    # PROPERTIES

    @property
    def pairs(self):
        """
        :return: pairs of similar documents
        """
        self._check_evaluated()
        return self._results[self.PAIRS_KEY]

    @property
    def execution_time(self):
        """
        :return: evaluation elapsed time
        """
        self._check_evaluated()
        return self._results[self.TIME_KEY]

    @property
    def score(self) -> float:
        """ Compare pairs with exact solution"""

        if not check_exact_evaluation(data_name=self._data_name):
            raise Exception("Exact solution was not computed. Use ExactSolutionEvaluation to compute and save it.")

        # loading exact solution
        file = get_exact_solution_file(data_name=self._data_name)
        exact_pairs = load_evaluation(path_=file)[self.PAIRS_KEY]
        exact_pairs = [(a, b) for a, b in exact_pairs]  # convert from lists of two in tuples

        actual_pairs = self.pairs

        # computing jaccard similarity
        return jaccard(s1=set(exact_pairs), s2=set(actual_pairs))

    # EVALUATION

    @abstractmethod
    def evaluate(self):
        """
        Evaluate the model
        This method should perform pairwise similarity evaluation and save results
            in the dictionary providing pairs found, level of similarity and elapsed time
        Finally it must set evaluated flag on
        """

        pass

    def _load_vectors(self) -> csr_matrix:
        """
        Load vectorized documents
        :return: matrix of vectorized documents
        """

        log(info="Loading vectors... ")
        file = get_vector_file(data_name=self._data_name)
        return load_vectors(path_=file)

    def _check_evaluated(self):
        """
        Check if computation was evaluated,
            raise an exception otherwise
        """
        if not self._evaluated:
            raise Exception("Evaluation not computed yet")

    # SAVE

    @abstractmethod
    def save(self, file_name: str = 'evaluation'):
        pass

    def _save(self, file_name: str):
        """
        Save pairs to disk
        :param file_name: name for output file
        """

        self._check_evaluated()

        log(info=f"Saving evaluation results. ")

        make_dir(get_evaluation_dir(data_name=self._data_name))

        file = get_evaluation_file(data_name=self._data_name, file_name=file_name)
        save_evaluation(eval_=self._results, path_=file)


class ExactSolutionEvaluation(SimilarityPairsEvaluation):
    """ This class compute exact solution """

    _CLASS_NAME = "ExactSolutionEvaluation"

    # DUNDER

    def __init__(self, data_name: str, threshold: float):
        """
        :param data_name: dataset name in datasets folder
        """

        super().__init__(data_name, threshold)

    # EVALUATION

    def evaluate(self):
        """
        Evaluate the model
        """

        log(info="Evaluating... ")

        t1 = time.perf_counter()

        vectors: csr_matrix = self._document_vectors.vectors

        pairs = []
        n_docs, _ = vectors.shape

        log(info="Computing cosine similarity... ")

        similarity_matrix = cosine_similarity(vectors)
        are_similar = similarity_matrix > self._threshold

        log(info="Inspecting similarities... ")

        for i, j in combinations(range(n_docs), 2):
            if are_similar[i, j]:
                id_1, _ = self._document_vectors.get_row_info(row=i)
                id_2, _ = self._document_vectors.get_row_info(row=j)
                pairs.append((id_1, id_2))

        t2 = time.perf_counter()

        self._results = {
            self.PAIRS_KEY: pairs,
            self.THRESHOLD_KEY: self._threshold,
            self.TIME_KEY: t2 - t1
        }

        self._evaluated = True

    # SAVE

    def save(self, file_name: str = 'evaluation'):
        """
        Save evaluation
        :param file_name: is ignored
        """
        self._save(file_name=EXACT_SOLUTION)


class DimensionalityHeuristicEvaluation(SimilarityPairsEvaluation):
    """ This class compute pairs using different heuristics:
        - dimensionality reduction
    """

    _CLASS_NAME = "DimensionalityHeuristicEvaluation"

    # DUNDER

    def __init__(self, data_name: str, threshold: float, eps: float):
        """
        :param data_name: dataset name in datasets folder
        :param eps: new dimensionality for vectors
        """

        super().__init__(data_name, threshold)

        # dim_reduction
        if not 0 < eps < 1:
            raise Exception(f"Invalid approximation error {eps}: not in range ]0, 1[ ")
        self._eps: float | None = eps

    def __str__(self):
        """
        :return: string representation for the object
        """
        return f"{super().__str__()} ['dim_reduction':  {self._eps} approx error] "

    # EVALUATION

    def evaluate(self):
        """
        Evaluate the model
        """

        # -- DIMENSIONALITY REDUCTION HEURISTIC --

        log(info="Performing dimensionality reduction... ")

        pt1 = time.perf_counter()

        self._document_vectors.perform_dimensionality_reduction(eps=self._eps)
        vectors = self._document_vectors.vectors_reduced

        pt2 = time.perf_counter()

        # ----------------------------------------

        t1 = time.perf_counter()

        n_docs, _ = vectors.shape

        log(info="Computing similarities... ")

        similarity_matrix = cosine_similarity(vectors)
        are_similar = similarity_matrix > self._threshold

        log(info="Inspecting similarities... ")

        pairs = []

        for i, j in combinations(range(n_docs), 2):
            if are_similar[i, j]:
                id_1, _ = self._document_vectors.get_row_info(row=i)
                id_2, _ = self._document_vectors.get_row_info(row=j)
                pairs.append((id_1, id_2))

        t2 = time.perf_counter()

        self._results = {
            self.PAIRS_KEY: pairs,
            self.THRESHOLD_KEY: self._threshold,
            self.TIME_KEY: t2 - t1
        }

        self._evaluated = True

    # SAVE

    def save(self, file_name: str = 'dim_reduction_heuristic'):
        """
        Save evaluation
        :param file_name: is ignored
        """
        self._save(file_name=file_name)


class DocTermsHeuristicEvaluation(SimilarityPairsEvaluation):
    """ This class compute pairs using different heuristics:
        - term size skip
    """

    _CLASS_NAME = "DocTermsHeuristicEvaluation"

    # DUNDER

    def __init__(self, data_name: str, threshold: float,
                 k: float | None = None):
        """
        :param data_name: dataset name in datasets folder
        :param k: doc-terms multiplication factor
        """

        super().__init__(data_name, threshold)

        if k < 1:
            raise Exception(f"Invalid mult factor {k}: not greater than 1 ")
        self._k: float | None = k

    def __str__(self):
        """
        :return: string representation for the object
        """
        return f"{super().__str__()} ['docs_size': {self._k} mult factor]"

    # EVALUATION

    def evaluate(self):
        """
        Evaluate the model
        """

        log(info="Evaluating... ")

        t1 = time.perf_counter()

        vectors: csr_matrix = self._document_vectors.vectors

        pairs = []
        n_docs, _ = vectors.shape

        log(info="Computing cosine similarity... ")

        similarity_matrix = cosine_similarity(vectors)
        are_similar = similarity_matrix > self._threshold

        log(info="Inspecting similarities... ")

        for i in range(n_docs - 1):

            id_1, len_1 = self._document_vectors.get_row_info(row=i)

            for j in range(i + 1, n_docs):

                id_2, len_2 = self._document_vectors.get_row_info(row=j)

                # -- DOC-TERMS HEURISTIC --
                if len_1 / len_2 > self._k:
                    break
                # ------------------------

                if are_similar[i, j]:
                    pairs.append((id_1, id_2))

        t2 = time.perf_counter()

        self._results = {
            self.PAIRS_KEY: pairs,
            self.THRESHOLD_KEY: self._threshold,
            self.TIME_KEY: t2 - t1
        }

        self._evaluated = True

    # SAVE

    def save(self, file_name: str = 'doc_size_heuristic'):
        """
        Save evaluation
        :param file_name: is ignored
        """
        self._save(file_name=file_name)

