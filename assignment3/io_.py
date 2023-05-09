"""
This module collects all operations related to input / output
"""
from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from os import path as path
from typing import Tuple, List, Dict

import pandas as pd
from pandas import DataFrame
from scipy import sparse
from scipy.sparse import csr_matrix

from assignment3.settings import LOG, DATASETS_DIR_NAME, CORPUS, QUERIES, TEST, VECTOR_DIR, VECTOR_FILE, IMAGES_DIR, EXACT_SOLUTION, \
    EVALUATION_DIR, VECTOR_MAPPING, VECTOR_INVERSE_MAPPING


# LOGGER

def log(info: str):
    """
    Log message if enabled from settings
    :param info: information to log
    """
    if LOG:
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
    return path.exists(get_dataset_dir(data_name=data_name)) and \
           path.exists(corpus) and path.exists(queries) and path.exists(test)


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
    :return: path to vector file
    """

    vector_dir = get_vector_dir(data_name=data_name)

    return path.join(vector_dir, f"{VECTOR_FILE}.npz")


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


def check_dataset_vectorized(data_name: str) -> bool:
    """
    Check if a certain dataset was vectorized
    :param data_name: name of dataset in datasets folder
    :return: true if dataset is vectorized, false otherwise
    """
    vector_file = get_vector_file(data_name=data_name)
    return path.exists(get_vector_dir(data_name=data_name)) and \
           path.exists(vector_file)


# EVALUATION

def get_evaluation_dir(data_name: str) -> str:
    """
    Return path to folder containing evaluation files
    :param data_name: name of dataset in datasets folder
    :return: path to evaluation folder
    """
    return path.join(get_dataset_dir(data_name=data_name), EVALUATION_DIR)


def get_exact_solution_file(data_name: str) -> str:
    """
    Return path to exact solution file
    :param data_name: name of dataset in datasets folder
    :return: path to exact solution file
    """

    evaluation_dir = get_evaluation_dir(data_name=data_name)

    return path.join(evaluation_dir, f"{EXACT_SOLUTION}.json")


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

def make_dir(path_: str) -> bool:
    """
    Create directory or ignore if it already exists
    :return: true if directory was created, false if it already existed
    """
    try:
        os.makedirs(path_)
        log(info=f"Creating directory {path_} ")
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
    log(info=f"Loading {path_} ")
    return pd.read_json(path_, lines=True).loc[:999]


# CSR MATRIX

def save_sparse_matrix(mat: csr_matrix, path_: str):
    """
    Save sparse matrix to disk
    :param mat: sparse matrix
    :param path_: local file path
    """
    log(info=f"Saving {path_} ")
    sparse.save_npz(file=path_, matrix=mat)


def load_sparse_matrix(path_: str) -> csr_matrix:
    """
    Load sparse matrix from disk
    :param path_: local file path
    :return: sparse matrix
    """
    log(info=f"Loading {path_} ")
    return sparse.load_npz(file=path_)


# PANDAS DATAFRAME

def save_dataframe(df: DataFrame, path_: str):
    """
    Save dataframe to disk
    :param df: dataframe
    :param path_: local file path
    """
    log(info=f"Saving {path_} ")
    df.to_csv(path_or_buf=path_)


def load_dataframe(path_: str) -> DataFrame:
    """
    Load dataframe from disk
    :param path_: local file path
    :return: dataframe
    """
    log(info=f"Loading {path_} ")
    return pd.read_csv(filepath_or_buffer=path_)


# JSON

def _write_json(obj: Dict, path_: str):
    """
    Save object to a json file
    :param obj: object to be stored
    :param path_: local file path
    """
    log(info=f"Saving {path_} ")
    with open(path_, 'w') as f:
        json.dump(obj, f)


def _load_json(path_: str) -> Dict:
    """
    Load object from disk
    :param path_: local file path
    :return: object
    """
    log(info=f"Loading {path_} ")

    with open(path_, 'r') as f:
        obj = json.load(f)
        return obj


def save_evaluation(eval_: Dict[Tuple[str, str]], path_: str):
    """
    Store list of pairs of integers to disk
    :param eval_: list of pairs
    :param path_: local file path
    """
    _write_json(obj=eval_, path_=path_)


def load_evaluation(path_: str) -> Dict:
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
    _write_json(obj=dict_, path_=path_)


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
    Store inverse vector mapping to disk
    :param dict_: mapping between doc_id and row
    :param path_: local file path
    """
    _write_json(obj=dict_, path_=path_)


def load_inverse_mapping(path_: str) -> Dict[str, int]:
    """
    Load inverse vector mapping from disk
    :param path_: local file path
    :return: mapping between doc_id and row
    """
    return _load_json(path_=path_)
