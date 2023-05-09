from __future__ import annotations

import time
from abc import ABC, abstractmethod
from itertools import combinations
from typing import Dict

import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jaccard
from sklearn.metrics.pairwise import cosine_similarity

from assignment3.io_ import log, get_vector_file, load_sparse_matrix, make_dir, get_evaluation_dir, \
    get_exact_solution_file, save_evaluation, load_evaluation, check_exact_evaluation, get_evaluation_file
from assignment3.model.documents import DocumentVectors
from assignment3.settings import EXACT_SOLUTION


class SimilarityPairsEvaluation(ABC):
    """ This class compute exact solution """

    TIME_KEY = 'execution_time'
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

    def __str__(self):
        """
        :return: string representation for the object
        """
        return f"{self._CLASS_NAME} - {self._data_name} ({self._threshold} similarity) "

    def __repr__(self):
        """
        :return: string representation for the object
        """
        return str(self)

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

        actual_pairs = set(self.pairs)

        # computing jaccard similarity
        intersection = len(actual_pairs.intersection(exact_pairs))
        union = len(actual_pairs.union(exact_pairs))

        return intersection / union

    # EVALUATION

    @abstractmethod
    def evaluate(self):
        """
        Evaluate the model
        """
        # TODO what it should do
        pass

    def _load_vectors(self) -> csr_matrix:
        """
        Load vectorized documents
        :return: matrix of vectorized documents
        """

        log(info="Loading vectors. ")
        file = get_vector_file(data_name=self._data_name)
        return load_sparse_matrix(path_=file)

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

        log(info="Evaluating. ")

        t1 = time.perf_counter()

        vectors: csr_matrix = self._document_vectors.vectors

        n_docs, _ = vectors.shape

        cos_sim_matrix = cosine_similarity(vectors)
        are_similar = cos_sim_matrix > self._threshold

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

    def save(self, file_name: str = 'evaluation'):
        """
        Save evaluation
        :param file_name: is ignored
        """
        self._save(file_name=EXACT_SOLUTION)


class HeuristicSolution(SimilarityPairsEvaluation):
    """ This class compute pairs using different heuristics:
        - dimensionality reduction
    """

    _CLASS_NAME = "HeuristicSolution"

    # DUNDER

    def __init__(self, data_name: str, threshold: float, new_dim: int | None = None):
        """
        :param data_name: dataset name in datasets folder
        :param new_dim: new dimensionality for vectors, if not given original one is preserved
        """

        super().__init__(data_name, threshold)

        # heuristics param, if None heuristic is not used
        self._new_dim: int | None = new_dim

    def __str__(self):
        """
        :return: string representation for the object
        """
        f"{super(HeuristicSolution, self).__str__()} [Heuristics: 'dim_reduction':  {self._new_dim}]"

    # EVALUATION

    def evaluate(self):
        """
        Evaluate the model
        """

        log(info="Evaluating. ")

        t1 = time.perf_counter()

        # DIMENSIONALITY REDUCTION HEURISTIC

        vectors: csr_matrix | np.ndarray

        if self._new_dim is None:
            vectors = self._document_vectors.vectors
        else:
            self._document_vectors.perform_dimensionality_reduction(new_dim=self._new_dim)
            vectors = self._document_vectors.vectors_reduced

        n_docs, _ = vectors.shape

        cos_sim_matrix = cosine_similarity(vectors)
        are_similar = cos_sim_matrix > self._threshold

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

    def save(self, file_name: str = 'evaluation'):
        """
        Save evaluation
        :param file_name: is ignored
        """
        self._save(file_name=EXACT_SOLUTION)
