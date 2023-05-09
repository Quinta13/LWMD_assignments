from __future__ import annotations

import time
from itertools import combinations
from typing import List, Tuple, Dict

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from assignment3.io_ import log, get_vector_file, load_sparse_matrix, make_dir, get_evaluation_dir, get_exact_solution_file, \
    save_evaluation
from assignment3.model.documents import DocumentVectors


class ExactSolutionEvaluation:
    """ This class compute exact solution """

    _TIME_KEY = 'execution_time'
    _THRESHOLD_KEY = 'threshold'
    _PAIRS_KEY = 'pairs'

    # DUNDER

    def __init__(self, data_name: str):
        """
        :param data_name: dataset name in datasets folder
        """

        self._data_name = data_name

        self._document_vectors: DocumentVectors =\
            DocumentVectors(data_name=self._data_name)  # delegate on download and vectorization check

        self._results: Dict = dict()  # available after evaluation
        self._evaluated: bool = False

    def __str__(self):
        """
        :return: string representation for the object
        """
        return f"ExactSolutionComputation - {self._data_name} "

    # PROPERTIES

    @property
    def pairs(self):
        """
        :return: pairs of similar documents
        """
        self._check_evaluated()
        return self._results[self._PAIRS_KEY]

    @property
    def execution_time(self):
        """
        :return: evaluation elapsed time
        """
        self._check_evaluated()
        return self._results[self._TIME_KEY]

    # EVALUATION

    def evaluate(self, threshold: float):
        """
        Evaluate the model
        :param threshold: similarity threshold
        """

        if not 0 <= threshold <= 1:
            raise Exception(f"Invalid threshold {threshold}: not in range [0, 1] ")

        log(info="Evaluating. ")

        t1 = time.perf_counter()

        vectors: csr_matrix = self._document_vectors.vectors

        n_docs, _ = vectors.shape

        cos_sim_matrix = cosine_similarity(vectors)
        are_similar = cos_sim_matrix > threshold

        pairs = []

        for i, j in combinations(range(n_docs), 2):
            if are_similar[i, j]:
                id_1, _ = self._document_vectors.get_row_info(row=i)
                id_2, _ = self._document_vectors.get_row_info(row=j)
                pairs.append((id_1, id_2))

        t2 = time.perf_counter()

        self._results = {
            self._PAIRS_KEY: pairs,
            self._THRESHOLD_KEY:  threshold,
            self._TIME_KEY: t2 - t1
        }

        self._evaluated = True

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

    def save(self):
        """
        Save pairs to disk
        """

        self._check_evaluated()

        log(info=f"Saving pairs. ")

        make_dir(get_evaluation_dir(data_name=self._data_name))

        file = get_exact_solution_file(data_name=self._data_name)
        save_evaluation(eval_=self._results, path_=file)
