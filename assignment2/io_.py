"""
This module collects all operations related to input / output
"""

import os
import re
import urllib.request
import zipfile
from os import path as path
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix

from assignment2.settings import DATASETS_DIR_NAME, CORPUS, QUERIES, TEST, VECTOR_DIR, SPARSE_DOCS, SPARSE_QUERY, \
    DENSE_DOCS, \
    DENSE_QUERY, SCORE_DIR, SPARSE_SCORES, FULL_SCORES, DENSE_SCORES, EVALUATION_DIR, IMAGES_DIR_NAME, EVALUATION_FILE, \
    LOG


# LOGGER
def _log(info: str):
    """
    Log message if enabled from settings
    :param info: information to log
    """
    _log(info)


# INPUT DIRECTORY and FILES

def get_root_dir() -> str:
    """
    Returns the path to the root directory project.
    :return: string representing the dir path
    """

    # Remember that the relative path here is relative to __file__,
    # so an additional ".." is needed
    return str(path.abspath(path.join(__file__, "../")))


def get_dataset_dir() -> str:
    """
    Returns the path to the dataset directory.
    :return: string representing the dataset directory
    """

    return path.join(get_root_dir(), DATASETS_DIR_NAME)


def get_files(data_name: str) -> Tuple[str, str, str]:
    """
    Return path to files constituting a dataset
    :param data_name: name of dataset in datasets folder
    :return: path to corpus file, queries file and test file
    """
    return path.join(get_dataset_dir(), data_name, CORPUS), \
           path.join(get_dataset_dir(), data_name, QUERIES), \
           path.join(get_dataset_dir(), data_name, TEST)


# VECTORIZATION DIRECTORY and FILES

def get_vector_dir(data_name: str) -> str:
    """
    Return path to folder containing vectorized documents
    :param data_name: name of dataset in datasets folder
    :return: path to vector folder
    """
    return path.join(get_dataset_dir(), data_name, VECTOR_DIR)


def get_sparse_vector_files(data_name: str) -> Tuple[str, str]:
    """
    Return path to sparse vectorized files
    :param data_name: name of dataset in datasets folder
    :return: path to sparse document and sparse query files
    """

    vector_dir = get_vector_dir(data_name=data_name)

    return path.join(vector_dir, f"{SPARSE_DOCS}.npz"), \
           path.join(vector_dir, f"{SPARSE_QUERY}.npz")


def get_dense_vector_files(data_name: str) -> Tuple[str, str]:
    """
    Return path to dense vectorized files
    :param data_name: name of dataset in datasets folder
    :return: path to dense document and dense query files
    """

    vector_dir = get_vector_dir(data_name=data_name)

    return path.join(vector_dir, f"{DENSE_DOCS}.npy"), \
           path.join(vector_dir, f"{DENSE_QUERY}.npy")


# SCORE DIRECTORY and FILES

def get_scores_dir(data_name: str) -> str:
    """
    Return path to folder containing scores
    :param data_name: name of dataset in datasets folder
    :return: path to scores folder
    """
    return path.join(get_dataset_dir(), data_name, SCORE_DIR)


def get_scores_files(data_name: str) -> Tuple[str, str, str]:
    """
    Return file paths to scores files
    :param data_name: name of dataset in datasets folder
    :return: path to sparse, dense and full scores files
    """

    score_dir = get_scores_dir(data_name=data_name)

    return path.join(score_dir, f"{SPARSE_SCORES}.npy"), \
           path.join(score_dir, f"{DENSE_SCORES}.npy"), \
           path.join(score_dir, f"{FULL_SCORES}.npy")


# EVALUATION DIRECTORY and FILES

def get_evaluation_dir(data_name: str) -> str:
    """
    Return path to folder containing evaluation results
    :param data_name: name of dataset in datasets folder
    :return: path to evaluation folder
    """
    return path.join(get_dataset_dir(), data_name, EVALUATION_DIR)


def get_evaluation_file(data_name: str, k: int) -> str:
    """
    Return single evaluation file
    :param data_name: name of dataset in datasets folder
    :param k: k used for evaluation
    :return: path to evaluation_files
    """

    evaluation_dir = get_evaluation_dir(data_name=data_name)
    out_file_name = f"{EVALUATION_FILE}_k{k}.csv"
    out_file = path.join(evaluation_dir, out_file_name)

    return out_file


def get_evaluation_files(data_name: str) -> List[Tuple[int, pd.DataFrame]]:
    """
    Return list of evaluation files
    :param data_name: name of dataset in datasets folder
    :return: list of tuples (k, dataframe)
    """

    dir_ = get_evaluation_dir(data_name=data_name)

    pattern = r'^evaluation_k(\d+)\.csv$'

    out = []

    for file_name in os.listdir(dir_):
        match = re.match(pattern=pattern, string=file_name)
        if match:
            number = int(match.group(1))
            file_name = get_evaluation_file(data_name=data_name, k=number)
            df = load_dataframe(path_=file_name)
            out.append((number, df))

    out = sorted(out)  # sort by k
    return out


# IMAGE DIRECTORY

def get_images_dir(data_name: str) -> str:
    """
    Return path to folder containing images results
    :param data_name: dataset name in datasets folder
    :return: path to images folder
    """
    return path.join(get_dataset_dir(), data_name, IMAGES_DIR_NAME)


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

def make_dir(path_: str):
    """
    Create directory or ignore if it already exists
    """
    try:
        os.makedirs(path_)
        _log(f"Creating directory {path_}")
    except OSError:
        pass  # ignore if already exists


# JSONL

def read_jsonl(path_: str) -> pd.DataFrame:
    """
    Parse .jsonl file into a pandas dataframe
    :param path_: path to .json file
    :return: .jsonl file content as a dataframe
    """
    _log(f"Loading {path_}")
    return pd.read_json(path_, lines=True)


# CSR MATRIX

def save_sparse_matrix(mat: csr_matrix, path_: str):
    """
    Save sparse matrix to disk
    :param mat: sparse matrix
    :param path_: local file path
    """
    _log(f"Saving {path_}")
    sparse.save_npz(file=path_, matrix=mat)


def load_sparse_matrix(path_: str) -> csr_matrix:
    """
    Load sparse matrix from disk
    :param path_: local file path
    :return: sparse matrix
    """
    _log(f"Loading {path_}")
    return sparse.load_npz(file=path_)


# NUMPY ARRAY

def save_dense_matrix(mat: np.ndarray, path_: str):
    """
    Save dense matrix to disk
    :param mat: dense matrix
    :param path_: local file path
    """
    _log(f"Saving {path_}")
    np.save(file=path_, arr=mat)


def load_dense_matrix(path_: str) -> np.ndarray:
    """
    Load dense matrix from disk
    :param path_: local file path
    :return: dense matrix
    """
    _log(f"Loading {path_}")
    return np.load(file=path_)


# DATAFRAME

def save_dataframe(df: pd.DataFrame, path_: str):
    """
    Save dataframe to disk
    :param df: dataframe
    :param path_: local file path
    """
    _log(f"Saving {path_}")
    df.to_csv(path_, index=False)


def load_dataframe(path_: str) -> pd.DataFrame:
    """
    Load dataframe from disk
    :param path_: local file path
    :return: dataframe
    """
    _log(f"Loading {path_}")
    return pd.read_csv(path_)
