"""
This module collects all operations related to input / output
"""
from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from os import path as path
from typing import Tuple, Dict, List, Any

import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix

from assignment3.settings import LOG, DATASETS_DIR_NAME, CORPUS, QUERIES, TEST, VECTOR_DIR, IMAGES_DIR, \
    EXACT_SOLUTION, \
    EVALUATION_DIR, VECTOR_MAPPING, VECTOR_INVERSE_MAPPING, IO_LOG, VECTOR_FILE, IDF_PERMUTATION, SKETCHING_DIR, \
    SKETCHING_FILE, TERMS_INFO, TERMS_INFO_IDF


# LOGGER

def log(info: str):
    """
    Log message if enabled from settings
    :param info: information to log
    """
    if LOG:
        print(info)


def _io_log(info: str):
    """
    Log message if enabled from settings
    :param info: information to log
    """
    if IO_LOG:
        print(info)


# INPUT DIRECTORY and FILES

def get_root_dir() -> str:
    """
    Returns the path to the root directory project.
    :return: string representing the dir path
    """

    # Remember that the relative path here is relative to __file__,
    # so an additional ".." is needed
    return str(path.abspath(path.join(__file__, "../")))


def get_dataset_main_dir() -> str:
    """
    Returns the path to the dataset directory.
    :return: string representing the dataset directory
    """

    return path.join(get_root_dir(), DATASETS_DIR_NAME)


def get_dataset_dir(data_name: str) -> str:
    """
    Returns path to a specific data directory in datasets folder
    :param data_name: name of datasets in datasets folder
    :return: path to dataset
    """

    return path.join(get_dataset_main_dir(), data_name)


def get_files(data_name: str) -> Tuple[str, str, str]:
    """
    Return path to files constituting a dataset
    :param data_name: name of dataset in datasets folder
    :return: path to corpus file, queries file and test file
    """
    return path.join(get_dataset_dir(data_name=data_name), CORPUS), \
        path.join(get_dataset_dir(data_name=data_name), QUERIES), \
        path.join(get_dataset_dir(data_name=data_name), TEST)


def check_dataset_downloaded(data_name: str) -> bool:
    """
    Check if a certain dataset was downloaded with its own files
    :param data_name: name of dataset in datasets folder
    :return: true if dataset is already present, false otherwise
    """
    corpus, queries, test = get_files(data_name=data_name)
    return path.exists(path=get_dataset_dir(data_name=data_name)) and \
        path.exists(path=corpus) and path.exists(path=queries) and path.exists(path=test)


# VECTORIZATION DIRECTORY and FILE

def get_vector_dir(data_name: str) -> str:
    """
    Return path to folder containing vectorized documents
    :param data_name: name of dataset in datasets folder
    :return: path to vector folder
    """
    return path.join(get_dataset_dir(data_name=data_name), VECTOR_DIR)


def get_vector_file(data_name: str) -> str:
    """
    Return path to vector file
    :param data_name: name of dataset in datasets folder
    :param vector_name: name of vector file
    :return: path to vector file
    """

    vector_dir = get_vector_dir(data_name=data_name)

    return path.join(vector_dir, f"{VECTOR_FILE}.npz")


def get_idf_permutation_file(data_name: str) -> str:
    """
    Return path to idf permutation file
    :param data_name: name of dataset in datasets folder
    :return: path to idf permutation file
    """

    vector_dir = get_vector_dir(data_name=data_name)

    return path.join(vector_dir, f"{IDF_PERMUTATION}.json")


def get_mapping_file(data_name: str) -> str:
    """
    Return path to vector mapping file
    :param data_name: name of dataset in datasets folder
    :return: path to vector mapping file
    """

    vector_dir = get_vector_dir(data_name=data_name)

    return path.join(vector_dir, f"{VECTOR_MAPPING}.json")


def get_inverse_mapping_file(data_name: str) -> str:
    """
    Return path to vector inverse mapping file
    :param data_name: name of dataset in datasets folder
    :return: path to vector inverse mapping file
    """
    vector_dir = get_vector_dir(data_name=data_name)

    return path.join(vector_dir, f"{VECTOR_INVERSE_MAPPING}.json")


def get_inverse_mapping_file(data_name: str) -> str:
    """
    Return path to vector inverse mapping file
    :param data_name: name of dataset in datasets folder
    :return: path to vector inverse mapping file
    """
    vector_dir = get_vector_dir(data_name=data_name)

    return path.join(vector_dir, f"{VECTOR_INVERSE_MAPPING}.json")


def get_terms_info_file(data_name: str) -> str:
    """
    Return path to terms info file
    :param data_name: name of dataset in datasets folder
    :return: path to vector terms info file
    """

    vector_dir = get_vector_dir(data_name=data_name)

    return path.join(vector_dir, f"{TERMS_INFO}.json")


def get_terms_info_idf_file(data_name: str) -> str:
    """
    Return path to terms info idf file
    :param data_name: name of datasterms info idf vector inverse file
    """

    vector_dir = get_vector_dir(data_name=data_name)

    return path.join(vector_dir, f"{TERMS_INFO_IDF}.json")


def check_dataset_vectorized(data_name: str) -> bool:
    """
    Check if a certain dataset was vectorized
    :param data_name: name of dataset in datasets folder
    :return: true if dataset is vectorized, false otherwise
    """
    vector_file = get_vector_file(data_name=data_name)

    permutation = get_idf_permutation_file(data_name=data_name)
    mapping = get_mapping_file(data_name=data_name)
    inverse_mapping = get_inverse_mapping_file(data_name=data_name)
    terms_info = get_terms_info_file(data_name=data_name)
    terms_info_idf = get_terms_info_idf_file(data_name=data_name)

    return path.exists(path=get_vector_dir(data_name=data_name)) and \
        path.exists(path=vector_file) and \
        path.exists(path=permutation) and \
        path.exists(path=mapping) and \
        path.exists(path=inverse_mapping) and\
        path.exists(path=terms_info) and \
        path.exists(path=terms_info_idf)


# EVALUATION

def get_evaluation_dir(data_name: str) -> str:
    """
    Return path to folder containing evaluation files
    :param data_name: name of dataset in datasets folder
    :return: path to evaluation folder
    """
    return path.join(get_dataset_dir(data_name=data_name), EVALUATION_DIR)


def get_evaluation_file(data_name: str, file_name: str) -> str:
    """
    Return path to exact solution file
    :param data_name: name of dataset in datasets folder
    :param file_name: name for evaluation file
    :return: path to exact solution file
    """

    evaluation_dir = get_evaluation_dir(data_name=data_name)

    return path.join(evaluation_dir, f"{file_name}.json")


def get_exact_solution_file(data_name: str) -> str:
    """
    Return path to exact solution file
    :param data_name: name of dataset in datasets folder
    :return: path to exact solution file
    """

    return get_evaluation_file(data_name=data_name, file_name=EXACT_SOLUTION)


def check_exact_evaluation(data_name: str) -> bool:
    """
    Check if a exact evaluation for a given dataset was computed
    :param data_name: name of dataset in datasets folder
    :return: true if exact solution was computed, false otherwise
    """
    exact_solution_file = get_exact_solution_file(data_name=data_name)
    return path.exists(path=get_evaluation_dir(data_name=data_name)) and \
        path.exists(path=exact_solution_file)


# SKETCHING

def get_sketching_dir(data_name: str) -> str:
    """
    Return path to folder containing sketching files
    :param data_name: name of dataset in datasets folder
    :return: path to sketching folder
    """
    return path.join(get_dataset_dir(data_name=data_name), SKETCHING_DIR)


def get_signatures_file(data_name: str) -> str:
    """
    Return path to signatures file
    :param data_name: name of dataset in datasets folder
    :return: path to exact signatures file
    """

    evaluation_dir = get_sketching_dir(data_name=data_name)

    return path.join(evaluation_dir, f"{SKETCHING_FILE}.json")


def check_sketching(data_name: str) -> bool:
    """
    Check if a sketching for a given dataset was computed
    :param data_name: name of dataset in datasets folder
    :return: true if sketching was computed, false otherwise
    """
    sketch_file = get_signatures_file(data_name=data_name)
    return path.exists(path=get_sketching_dir(data_name=data_name)) and \
        path.exists(path=sketch_file)


# IMAGE DIRECTORY

def get_images_dir(data_name: str) -> str:
    """
    Return path to folder containing images results
    :param data_name: dataset name in datasets folder
    :return: path to images folder
    """
    return path.join(get_dataset_main_dir(), data_name, IMAGES_DIR)


# DOWNLOAD

def download_and_unzip(url: str, output_dir: str):
    """
    Download and unzip datasets from url
    :param url: url of compressed file
    :param output_dir: directory where to store dataset
    """

    # Download the zip file
    zip_file_name = url.split("/")[-1]
    zip_file_path = os.path.join(output_dir, zip_file_name)
    urllib.request.urlretrieve(url, zip_file_path)

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    # Delete the zip file
    os.remove(zip_file_path)

    # Remove the .zip extension from the extracted file
    extracted_file_name = zip_file_name.split(".")[0]
    extracted_file_path = os.path.join(output_dir, extracted_file_name)
    os.rename(os.path.join(output_dir, extracted_file_name), extracted_file_path)

    return extracted_file_path


# IO OPERATIONS for SPECIFIC FORMAT

def _check_extension(path_: str, ext: str):
    """
    Check if given path as certain extension
        raise an error in not correct
    :param path_: file path to file
    :param ext: file extension
    """

    actual_ext = os.path.splitext(path_)[1]

    if actual_ext != f'.{ext}':
        raise Exception(f"Given file: {path_}, but '.{ext}' extension was expected")


def make_dir(path_: str) -> bool:
    """
    Create directory or ignore if it already exists
    :return: true if directory was created, false if it already existed
    """
    try:
        os.makedirs(path_)
        _io_log(info=f"Creating directory {path_} ")
        return True
    except OSError:
        return False  # it already exists


# JSONL

def read_jsonl(path_: str) -> pd.DataFrame:
    """
    Parse .jsonl file into a pandas dataframe
    :param path_: path to .json file
    :return: .jsonl file content as a dataframe
    """

    _check_extension(path_=path_, ext='jsonl')

    _io_log(info=f"Loading {path_} ")
    return pd.read_json(path_, lines=True)


# CSR MATRIX

def save_vectors(mat: csr_matrix, path_: str):
    """
    Save sparse matrix to disk
    :param mat: sparse matrix
    :param path_: local file path
    """

    _check_extension(path_=path_, ext='npz')

    _io_log(info=f"Saving {path_} ")
    sparse.save_npz(file=path_, matrix=mat)


def load_vectors(path_: str, idf_order: bool = False) -> csr_matrix:
    """
    Load sparse matrix from disk
    :param path_: local file path
    :param idf_order: if sort columns by increasing idf
    :return: sparse matrix
    """

    _check_extension(path_=path_, ext='npz')

    _io_log(info=f"Loading {path_} ")
    vectors = sparse.load_npz(file=path_)

    if idf_order:
        # sort columns by increasing idf
        data_name = path.basename(path.dirname(path.dirname(path_)))
        idf_permutation_file = get_idf_permutation_file(data_name=data_name)
        idf_permutation = _load_idf_permutation(path_=idf_permutation_file)
        vectors = vectors[:, idf_permutation]

    return vectors


# PANDAS DATAFRAME

# TODO remove if not used
"""

def save_dataframe(df: DataFrame, path_: str):
    \"""
    Save dataframe to disk
    :param df: dataframe
    :param path_: local file path
    \"""
    
    _check_extension(path_=path_, ext='csv')
    
    _io_log(info=f"Saving {path_} ")
    df.to_csv(path_or_buf=path_)


def load_dataframe(path_: str) -> DataFrame:
    \"""
    Load dataframe from disk
    :param path_: local file path
    :return: dataframe
    \"""
    
    _check_extension(path_=path_, ext='csv')

    _io_log(info=f"Loading {path_} ")
    return pd.read_csv(filepath_or_buffer=path_)
"""


# JSON

def _store_json(obj: Dict | List, path_: str):
    """
    Save object to a json file
    :param obj: object to be stored
    :param path_: local file path
    """

    _check_extension(path_=path_, ext='json')

    _io_log(info=f"Saving {path_} ")
    with open(path_, 'w') as f:
        json.dump(obj, f)


def _load_json(path_: str) -> Dict | List:
    """
    Load object from disk
    :param path_: local file path
    :return: object
    """
    _io_log(info=f"Loading {path_} ")

    _check_extension(path_=path_, ext='json')

    with open(path_, 'r') as f:
        obj = json.load(f)
        return obj


def save_evaluation(eval_: Dict[str, Any], path_: str):
    """
    Store list of pairs of integers to disk
    :param eval_: list of pairs
    :param path_: local file path
    """
    _store_json(obj=eval_, path_=path_)


def load_evaluation(path_: str) -> Dict[str, Any]:
    """
    Load list of pairs of integers from disk
    :param path_: local file path
    :return: list of pairs
    """
    return _load_json(path_=path_)


def save_mapping(dict_: Dict[int, Tuple[str, int]], path_: str):
    """
    Store vector mapping to disk
    :param dict_: mapping between row index and doc_id, terms
    :param path_: local file path
    """
    _store_json(obj=dict_, path_=path_)


def load_mapping(path_: str) -> Dict[int, Tuple[str, int]]:
    """
    Load vector mapping from disk
    :param path_: local file path
    :return: mapping between row index and doc_id, terms
    """
    mapping = _load_json(path_=path_)
    return {int(k): v for k, v in mapping.items()}  # cast row_idx to integer


def save_inverse_mapping(dict_: Dict[str, int], path_: str):
    """
     inverse vector mapping to disk
    :param dict_: mapping between doc_id and row
    :param path_: local file path
    """
    _store_json(obj=dict_, path_=path_)


def load_inverse_mapping(path_: str) -> Dict[str, int]:
    """
    Load inverse vector mapping from disk
    :param path_: local file path
    :return: mapping between doc_id and row
    """
    return _load_json(path_=path_)


def save_idf_permutation(list_: List[int], path_: str):
    """
    Store idf column permutation to disk
    :param list_: idf permutation
    :param path_: local file path
    """
    list_str = [str(i) for i in list_]
    _store_json(obj=list_str, path_=path_)


def _load_idf_permutation(path_: str) -> List[int]:
    """
    Load idf column permutation from disk
    :param path_: local file path
    :return: idf permutation
    """
    list_str = _load_json(path_=path_)
    return [int(i) for i in list_str]


def save_signatures(dict_: Dict[str, List[int]], path_: str):
    """
    Save sketching signatures to disk
    :param dict_: signatures
    :param path_: local file path
    """
    dict_str = {k: [str(i) for i in v] for k, v in dict_.items()}
    _store_json(obj=dict_str, path_=path_)


def load_signatures(path_: str) -> Dict[str, List[int]]:
    """
    Load sketching signatures from disk
    :param path_: local file path
    :return: signatures
    """
    dict_ = _load_json(path_=path_)
    return {k: [int(i) for i in v] for k, v in dict_.items()}


def save_terms_info(dict_: Dict[int, int], path_: str):
    """
    Save terms information to disk
    :param dict_: terms info
    :param path_: local file path
    """
    dict_str = {str(k): int(v) for k, v in dict_.items()}
    print(dict_str)
    _store_json(obj=dict_str, path_=path_)


def load_terms_info(path_: str) -> Dict[int, int]:
    """
    Load terms information from disk
    :padict_ = _load_json(path_=path_)
    return {k: int(v) for k, v in dict_.items()}ram path_: local file path
    :return: terms info
    """
    dict_ = _load_json(path_=path_)
    return {int(k): v for k, v in dict_.items()}
