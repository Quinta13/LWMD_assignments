"""
This module contains all classes part of the framework pipline
    - BEIRDatasetDownloader: dataset to download BEIR dataset
    - DocumentVectorizer: class uses for generate sparse and dense vectorization
"""

from __future__ import annotations

from typing import List, Dict, Tuple
from urllib.error import HTTPError

import numpy as np
from datasketch import MinHash
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from assignment3.io_ import get_dataset_main_dir, make_dir, download_and_unzip, log, check_dataset_downloaded, \
    get_vector_dir, save_vectors, get_mapping_file, \
    save_mapping, get_inverse_mapping_file, save_inverse_mapping, get_vector_file, get_idf_permutation_file, \
    save_idf_permutation, get_sketching_dir, get_signatures_file, save_signatures, get_terms_info_file, save_terms_info, \
    get_terms_info_idf_file
from assignment3.model.documents import DocumentsCollection
from assignment3.settings import DEFAULT_LANGUAGE, SIMILARITY


class BEIRDatasetDownloader:
    """
    This class allows to download BEIR datasets
    Dataset is stored in datasets folder; directory's name is the same as the datset
    """

    _URL_PREFIX = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/"  # prefix url for BEIR datasets

    # DUNDER

    def __init__(self, data_name: str):
        """
        :param data_name: dataset name
        """

        self._data_name = data_name

    def __str__(self):
        """
        :return: string representation of the object
        """
        return f"BEIRDatasetDownloader {self._data_name}"

    def __repr__(self):
        """
        :return: string representation of the object
        """
        return str(self)

    # PROPERTIES

    @property
    def is_downloaded(self) -> bool:
        """
        :return: is dataset is downloaded
        """
        return check_dataset_downloaded(data_name=self._data_name)

    # DOWNLOAD

    def download(self):
        """
        Download dataset in proper directory
            if it already downloaded it does nothing
        """

        url = f"{self._URL_PREFIX}{self._data_name}.zip"
        output_dir = get_dataset_main_dir()

        if not self.is_downloaded:
            make_dir(output_dir)
            log(info=f"Downloading {self._data_name} dataset. ")
            try:
                download_and_unzip(url=url, output_dir=output_dir)
            except HTTPError:
                raise Exception(f"Unable to fetch {self._data_name} dataset. "
                                "See https://github.com/beir-cellar/beir for valid dataset names")
        else:
            log(info=f"{self._data_name} dataset already exists. ")


class DocumentsVectorizer:
    """
    This class produce a vector representation of given dataset
        it access files basing on BEIR Datasets standard in terms of format and file organization
    """

    _ID_FIELD = '_id'
    _CONTENT_FIELD = 'text'

    def __init__(self, data_name: str, language: str = DEFAULT_LANGUAGE):
        """

        :param data_name: dataset name in datasets folder
        """

        self._data_name = data_name
        self._language = language

        self._documents: DocumentsCollection =\
            DocumentsCollection(data_name=self._data_name, language=self._language)  # check if dataset exists
        self._full_tokens: List[List[str]] = self._tokenize()

        self._document_vectors: csr_matrix = csr_matrix([])  # available after vectorization
        self._idf_permutation: np.ndarray = np.array([])     # available after vectorization
        self._mapping: Dict[int, Tuple[str, int]] = dict()   # available after vectorization
        self._inverse_mapping: Dict[str, int] = dict()       # available after vectorization
        self._term_info: Dict[int, int] = dict()             # available after vectorization
        self._term_info_idf: Dict[int, int] = dict()         # available after vectorization
        self._vectorized: bool = False

    def __str__(self):
        """
        :return: string representation for the object
        """
        return f"{self._data_name} Vectorizer [{len(self._documents)} documents]"

    def __repr__(self):
        """
        :return: string representation for the object
        """
        return str(self)

    # PROPERTIES

    @property
    def document_vectors(self) -> csr_matrix:
        """
        :return: sparse document vectors
        """
        self._check_vectorized()
        return self._document_vectors

    @property
    def document_vectors_idf(self) -> csr_matrix:
        """
        :return: sparse document vectors
        """
        self._check_vectorized()
        vectors = self.document_vectors
        return vectors[:, self._idf_permutation]

    @property
    def mapping(self) -> Dict[int, Tuple[str, int]]:
        """
        :return: mapping
        """
        self._check_vectorized()
        return self._mapping

    @property
    def inverse_mapping(self) -> Dict[str, int]:
        """
        :return: mapping
        """
        self._check_vectorized()
        return self._inverse_mapping

    # TOKENIZATION

    def _tokenize(self) -> List[List[str]]:
        """
        Tokenize documents, set on proper flag
        """

        log(info="Tokenizing documents. ")

        [doc.tokenize() for doc in self._documents]  # invoke tokenization for each document
        docs_full_tokens = [doc.tokens for doc in self._documents]
        return docs_full_tokens

    # VECTORIZATION

    def vectorize(self):
        """
        Create vector representation for documents
        """

        if self._vectorized:

            log(info="Documents were already vectorized ")
            return

        corpus: List[str] = [doc.tokenized_content for doc in self._documents if not doc.is_empty]

        log(info="Learning vocabulary idf. ")

        tfidf_vectorizer: TfidfVectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(raw_documents=corpus)

        self._idf_permutation = np.argsort(tfidf_vectorizer.idf_)

        log(info="Generating vector. ")

        self._document_vectors = tfidf_vectorizer.transform(corpus)

        log(info="Generating tf-idf mapper")

        log(info="Computing document length")

        # dictionary doc-id : # term
        doc_id_term_count = {
            doc_id : self._documents.get_document(id_=doc_id).distinct_terms
            for doc_id in self._documents.doc_ids
            if not self._documents.get_document(id_=doc_id).is_empty
        }
        sorted_doc_id_term_count = dict(sorted(doc_id_term_count.items(), key=lambda x: x[1], reverse=True))

        log(info="Generating permutation index")

        original_doc_ids = [doc.id_ for doc in self._documents if not doc.is_empty]
        ordered_doc_ids = list(sorted_doc_id_term_count.keys())

        # Mapping for ordered mapping
        index_map = {elem: index for index, elem in enumerate(ordered_doc_ids)}

        # Evaluating permutation indexes
        permutation_indexes = [index_map[elem] for elem in original_doc_ids]

        log(info="Permuting matrix")

        # get the sorted indices of permutation_indexes
        sorted_indices = np.argsort(permutation_indexes)

        # permute the rows of the document_vectors matrix
        self._document_vectors = self._document_vectors[sorted_indices, :]

        log(info="Generating mappings")

        self._mapping = {i: (key, sorted_doc_id_term_count[key]) for i, key in enumerate(sorted_doc_id_term_count)}
        self._inverse_mapping = {info[0] : row for row, info in self._mapping.items()}

        self._vectorized = True

        log(info="Computing terms information... ")

        self._term_info = self.get_relevant_terms(vectors=self._document_vectors, similarity=SIMILARITY)

        document_vector_idf = self._document_vectors[:, self._idf_permutation]
        self._term_info_idf = self.get_relevant_terms(vectors=document_vector_idf, similarity=SIMILARITY)

    @staticmethod
    def get_relevant_terms(vectors: csr_matrix, similarity: float) -> Dict[int, int]:
        """
        For each document provides term-id such that previous terms are sufficient
        :return:
        """

        def get_relevant_term(row: int, max_doc_: np.ndarray) -> int:

            doc = vectors[row]

            _, term_ids = doc.nonzero()  # terms in document
            term_ids.sort()

            k = term_ids[0]

            for k in term_ids:

                max_doc_first_k = max_doc_[:k+1]
                doc_first_k = doc[:, :k+1]
                sim = cosine_similarity(max_doc_first_k.reshape(1, -1), doc_first_k)[0][0]

                if sim > similarity:
                    break

            return k

        max_doc = vectors.max(axis=0).toarray().flatten()


        return {
            i : get_relevant_term(row=i, max_doc_=max_doc)
            for i in list(range(vectors.shape[0]))
        }

    def _check_vectorized(self):
        """
        Check if sparse vectorization was computed
        """
        if not self._vectorized:
            raise Exception("Documents were not vectorized yet")

    # SAVE

    def save(self):
        """
        Save sparse and dense representation to proper directory
            it may overwrite previous possible vectorizations
        """

        # exception delegated to the property
        docs = self.document_vectors

        make_dir(get_vector_dir(self._data_name))

        log(info="Saving vector")
        out_vect = get_vector_file(data_name=self._data_name)
        save_vectors(mat=docs, path_=out_vect)

        log(info="Saving idf permutation")
        idf_permut = get_idf_permutation_file(data_name=self._data_name)
        save_idf_permutation(list_=list(self._idf_permutation), path_=idf_permut)

        log(info="Saving mapping")
        out_map = get_mapping_file(data_name=self._data_name)
        save_mapping(dict_=self.mapping, path_=out_map)

        log(info="Saving inverse mapping")
        out_inverse_map = get_inverse_mapping_file(data_name=self._data_name)
        save_inverse_mapping(dict_=self.inverse_mapping, path_=out_inverse_map)

        log(info="Saving terms info")
        terms_info_file = get_terms_info_file(data_name=self._data_name)
        save_terms_info(dict_=self._term_info, path_=terms_info_file)

        terms_info_idf_file = get_terms_info_idf_file(data_name=self._data_name)
        save_terms_info(dict_=self._term_info_idf, path_=terms_info_idf_file)


class DocumentsSketching:
    """
    This class compute MinHash sketching between each sketch
    """

    # DUNDER

    def __init__(self, data_name: str, n_hash: int, language: str = DEFAULT_LANGUAGE):
        """
        :param data_name: name of dataset in datasets folder
        :param n_hash: number of hash functions
        :param language: document language
        """

        self._data_name: str = data_name
        self._n_hash: int = n_hash
        self._language: str = language

        self._documents: DocumentsCollection = \
            DocumentsCollection(data_name=self._data_name, language=self._language)  # check if dataset exists
        self._tokenize()

        self._signatures: Dict[str, List[int]] = dict()  # available after sketching
        self._sketched: bool = False

    def __str__(self):
        """
        :return: string representation for the object
        """
        return f"{self._data_name} Sketching [{len(self._documents)} docs, {self._n_hash} hash functions]"

    def __repr__(self):
        """
        :return: string representation for the object
        """
        return str(self)

    # PROPERTY

    @property
    def signatures(self) -> Dict[str, List[int]]:
        """
        :return: sketching
        """
        self._check_sketched()
        return self._signatures

    # TOKENIZATION

    def _tokenize(self):
        """
        Tokenizing documents
        """

        log(info="Tokenizing documents. ")

        [doc.tokenize() for doc in self._documents]  # side effect

    # SKETCHING

    def _minhash_signature(self, tokens: List[str]) -> List[int]:
        """
        Compute MinHash signature for one document
        :param tokens: document tokens
        """

        m = MinHash(self._n_hash)

        for token in tokens:
            m.update(token.encode('utf8'))

        return list(m.hashvalues)

    def sketch(self):
        """
        Compute signature for each document
        """

        log(info="Performing sketching")

        self._signatures = {
            doc.id_: self._minhash_signature(tokens=doc.tokens)
            for doc in self._documents
            if not doc.is_empty
        }

        self._sketched = True

    def _check_sketched(self):
        """
        Check if sketching was performed
            raise an error otherwise
        """
        if not self._sketched:
            raise Exception("Sketching not performed yet")

    # SAVE

    def save(self):

        """
        Save sparse and dense representation to proper directory
            it may overwrite previous possible vectorizations
        """

        # exception delegated to the property
        signatures = self.signatures

        make_dir(get_sketching_dir(self._data_name))

        log(info="Saving signatures")
        sign_file = get_signatures_file(data_name=self._data_name)
        save_signatures(dict_=signatures, path_=sign_file)
