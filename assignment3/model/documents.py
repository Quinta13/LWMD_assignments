"""
This file provide three classes to handle
- A single document
- A collection of document
- A collection of vectorized documents
"""

from __future__ import annotations

from collections import Counter
from typing import List, Iterator, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.random_projection import johnson_lindenstrauss_min_dim, SparseRandomProjection

from assignment3.io_ import check_dataset_downloaded, log, get_files, read_jsonl, check_dataset_vectorized, \
    load_vectors, get_mapping_file, load_mapping, get_inverse_mapping_file, load_inverse_mapping, get_vector_file, \
    get_terms_info_file, get_terms_info_idf_file, load_terms_info
from assignment3.settings import DEFAULT_LANGUAGE
from assignment3.utils import tokenize


class Document:
    """
    This class represent a document with its own doc-id and text
    """

    # DUNDER METHODS

    def __init__(self, id_: str, content: str, language: str = DEFAULT_LANGUAGE):
        """

        :param id_: doc-id
        :param content: document content
        :param language: language of the document (default in settings)
        """

        # constructor parameters
        self._id: str = id_
        self._content: str = content
        self._language: str = language

        self._tokens: List[str] = []  # not accessible before tokenization
        self._is_tokenized: bool = False

    def __str__(self) -> str:
        """
        :return: string representation for the object
        """
        return f"Document: {self.id_}"

    def __repr__(self) -> str:
        """
        :return: string representation for the object
        """
        return str(self)

    def __iter__(self) -> Iterator[str, str]:
        """
        :return: unpacked fields (doc_id, content)
        """
        return iter([self.id_, self.content])

    # PROPERTIES

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
        Return tokens of the document if it was tokenized
        :return: tokens
        """
        self._check_tokenized()
        return self._tokens

    @property
    def tokenized_content(self) -> str:
        """
        Return tokenized content
            full string it's not kept in memory but materialized using tokens
            so that the computation may require some additional time
        :return: tokenized content
        """
        # tokenization check is delegated to tokens property
        return " ".join(self.tokens)

    @property
    def tf(self) -> Dict[str, int]:
        """
        Returns terms frequencies in dictionary form
            terms are sorted in lexicographic order
        :return: document term frequency
        """
        # tokenization check is delegated to tokens property
        return dict(sorted(Counter(self.tokens).items()))

    @property
    def distinct_terms(self) -> int:
        """
        Returns number of distinct terms in the document
        :return: number of distinct terms in the document
        """
        # tokenization check is delegated to tf property
        return len(self.tf.keys())

    @property
    def is_empty(self) -> bool:
        """
        Returns if document content is empty
        """
        return len(self.tokens) == 0

    # TOKENIZATION

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
            raise Exception("Document not tokenized yet. Use Document.tokenize() method to do so. ")


class DocumentsCollection:
    """ This class provides some methods to use a collection of documents """

    # FIELDS BASED ON BEIR STANDARD
    _ID_FIELD = '_id'
    _CONTENT_FIELD = 'text'

    # DUNDER

    def __init__(self, data_name: str, language: str = DEFAULT_LANGUAGE):
        """

        :param data_name: dataset name in datasets folder
        """

        if not check_dataset_downloaded(data_name=data_name):
            raise Exception(f"{data_name} dataset and its files are not available. "
                            f"Use BEIRDatasetDownloader to download it.")

        self._data_name = data_name
        self._language = language

        self._documents: Dict[str, Document] = self._parse()

    def __len__(self):
        """
        :return: number of documents
        """
        return len(self._documents)

    def __str__(self):
        """
        :return: string representation for the object
        """
        return f"{self._data_name} Documents [{len(self)}]"

    def __repr__(self):
        """
        :return: string representation for the object
        """
        return str(self)

    def __iter__(self):
        """
        :return: iteration over documents
        """
        return iter(self._documents.values())

    # PROPERTY

    @property
    def doc_ids(self) -> np.ndarray:
        """
        :return: list of documents ids
        """
        return np.array(list(self._documents.keys()))

    # DOCUMENT INFO

    def get_document(self, id_: str) -> Document:
        """
        :param id_: document id
        :return: document with given id
        """
        try:
            return self._documents[id_]
        except KeyError:
            raise Exception(f"DocID: {id_} is not valid. ")

    def content_comparison(self, ids: Tuple[str, str]):
        """
        Print content of two given doc-ids in order to compare them
        :param ids: id of the two documents
        """

        for id_ in ids:
            log(info=f"Document {id_}: ")
            log(info=self.get_document(id_=id_).content)
            log(info="")

    # PARSING

    def _parse(self) -> Dict[str, Document]:
        """
        Parse documents
        :return: document collection as a dictionary
        """

        log(info="Parsing documents. ")

        file_, _, _ = get_files(self._data_name)

        docs_df: pd.DataFrame = read_jsonl(file_)

        documents: Dict[str, Document] = {
            row[self._ID_FIELD]:
                Document(id_=row[self._ID_FIELD], content=row[self._CONTENT_FIELD], language=self._language)
            for _, row in docs_df.iterrows()
        }

        return documents


class DocumentVectors:
    """ This class provide a vector view of the documents """

    def __init__(self, data_name: str, idf_order: bool = False):
        """

        :param data_name: name of dataset in datasets folder
        :param idf_order: if columns vectors are sorted by idf
        """

        self._data_name = data_name
        self._idf_order = idf_order

        if not check_dataset_vectorized(data_name=data_name):
            raise Exception(f"{data_name} dataset is not vectorized. "
                            f"Use BEIRDatasetVectorized to vectorize it. ")

        self._vectors_reduced: np.ndarray = np.array([])  # available after dimensionality reduction
        self._reduced: bool = False

        self._vectors: csr_matrix = self._load_vectors()
        self._mapping: Dict[int, Tuple[str, int]] = self._load_mapping()
        self._inverse_mapping: Dict[str, int] = self._load_inverse_mapping()

    def __len__(self):
        """
        :return: number of documents in the collection
        """
        return self._vectors.shape[0]

    def __str__(self):
        """
        :return: string representation for the object
        """
        return f"{self._data_name} Vector Documents [{len(self)}]"

    def __repr__(self):
        """
        :return: string representation for the object
        """
        return str(self)

    @property
    def vectors(self) -> csr_matrix:
        """
        :return: vector dataframe
        """
        return self._vectors

    @property
    def vectors_reduced(self) -> np.ndarray:
        """
        :return: vectors with reduced dimensionality
        """
        self._check_reduced()
        return self._vectors_reduced

    # MAPPING

    def get_row_info(self, row: int) -> Tuple[str, int]:
        """
        Return information about given row
        :param row: row index
        :return: information about doc_id and term count
        """
        return self._mapping[row]

    def get_doc_row(self, doc_id: str) -> int:
        """
        Return row of given doc-id
        :param doc_id: document id
        :return: information about doc_id and term count
        """
        return self._inverse_mapping[doc_id]

    # DIMENSIONALITY REDUCTION

    def perform_dimensionality_reduction(self, eps: float):
        """
        Reduce dataframe dimensionality with given approximation error
        :param eps: approximation error
        """

        # Compute the minimum number of components required to reduce the dimensionality
        n_components = johnson_lindenstrauss_min_dim(n_samples=self.vectors.shape[0], eps=eps)

        # Create a Sparse Random Projection object with the computed number of components
        transformer = SparseRandomProjection(n_components=n_components)

        # Fit and transform the original array
        self._vectors_reduced = transformer.fit_transform(self.vectors).toarray()

        self._reduced = True

    # LOADING

    def _load_vectors(self) -> csr_matrix:
        """
        Load vectorized documents
        :return: matrix of vectorized documents
        """

        log(info="Loading vectors. ")

        file = get_vector_file(data_name=self._data_name)
        mat: csr_matrix = load_vectors(path_=file, idf_order=self._idf_order)

        return mat

    def _load_mapping(self) -> Dict[int, Tuple[str, int]]:
        """
        Load mapping
        :return: mapping dictionary
        """

        log(info="Loading mapping. ")

        file = get_mapping_file(data_name=self._data_name)

        return load_mapping(path_=file)

    def _load_inverse_mapping(self) -> Dict[str, int]:
        """
        Load inverse mapping
        :return: inverse mapping dictionary
        """

        log(info="Loading inverse mapping. ")

        file = get_inverse_mapping_file(data_name=self._data_name)

        return load_inverse_mapping(path_=file)

    def _check_reduced(self):
        """
        Check if reduction was performed
        """
        if not self._reduced:
            raise Exception("Dimensionality reduction was not performed yet")

    # MAP REDUCE

    @property
    def documents_info(self) -> List[Tuple[str, int, List[int, float]]]:
        """
        Transpose vectors information in order to be processed by Mapper
        :return: list of tuples (doc-id ; list(term-id; value))
        """

        def extract_nonzero_entries(row: int) -> List[int, float]:
            """
            Return row mapping non-zero entries (term-id; entry value)
            :param row: row corresponding to certain document
            :return: list of tuples (term-id; entry value)
            """

            doc: csr_matrix = self.vectors[row]

            _, terms = doc.nonzero()
            terms = [int(t) for t in terms]
            terms.sort()

            entries = doc[0, terms].toarray().tolist()[0]

            return list(zip(terms, entries))

        terms_info_file: str = get_terms_info_idf_file(data_name=self._data_name) \
            if self._idf_order else \
            get_terms_info_file(data_name=self._data_name)

        terms_info: Dict[int, int] = load_terms_info(path_=terms_info_file)

        return [
            (
                self.get_row_info(row=index)[0],    # doc-id
                terms_info[index],                  # term-info
                extract_nonzero_entries(row=index)  # list(term-id, value)
            )
            for index in list(range(len(self)))  # row-id
        ]
