"""
Input - output operations
"""
import os
import re
from os import path as path
from typing import Tuple, List

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix

from settings import DATASETS_DIR_NAME, CORPUS, QUERIES, TEST, VECTOR_DIR, SPARSE_DOCS, SPARSE_QUERY, DENSE_DOCS, \
    DENSE_QUERY, SCORE_DIR, SPARSE_SCORES, FULL_SCORES, DENSE_SCORES, EVALUATION_DIR, IMAGES_DIR_NAME, EVALUATION_FILE, \
    LOG


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

    return path.join(vector_dir, f"{SPARSE_DOCS}.npz"),\
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

    return path.join(score_dir, f"{SPARSE_SCORES}.npy"),\
           path.join(score_dir, f"{DENSE_SCORES}.npy"),\
           path.join(score_dir, f"{FULL_SCORES}.npy")


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


def get_images_dir(data_name: str) -> str:
    """
    Return path to folder containing images results
    :param data_name: dataset name in datasets folder
    :return: path to images folder
    """
    return path.join(get_dataset_dir(), data_name, IMAGES_DIR_NAME)


def make_dir(path_: str):
    """
    Create directory or ignore if it already exists
    """
    try:
        os.makedirs(path_)
        if LOG:
            print(f"Creating directory {path_}")
    except OSError:
        pass  # ignore if already exists


def read_jsonl(path_: str) -> pd.DataFrame:
    """
    Parse .jsonl file into a pandas dataframe
    :param path_: path to .json file
    :return: .jsonl file content as a dataframe
    """
    if LOG:
        print(f"Loading {path_}")
    return pd.read_json(path_, lines=True)


def save_sparse_matrix(mat: csr_matrix, path_: str):
    """
    Save sparse matrix to disk
    :param mat: sparse matrix
    :param path_: local file path
    """
    if LOG:
        print(f"Saving {path_}")
    sparse.save_npz(file=path_, matrix=mat)


def load_sparse_matrix(path_: str) -> csr_matrix:
    """
    Load sparse matrix from disk
    :param path_: local file path
    :return: sparse matrix
    """
    if LOG:
        print(f"Loading {path_}")
    return sparse.load_npz(file=path_)


def save_dense_matrix(mat: np.ndarray, path_: str):
    """
    Save dense matrix to disk
    :param mat: dense matrix
    :param path_: local file path
    """
    if LOG:
        print(f"Saving {path_}")
    np.save(file=path_, arr=mat)


def load_dense_matrix(path_: str) -> np.ndarray:
    """
    Load dense matrix from disk
    :param path_: local file path
    :return: dense matrix
    """
    if LOG:
        print(f"Loading {path_}")
    return np.load(file=path_)


def save_dataframe(df: pd.DataFrame, path_: str):
    """
    Save dataframe to disk
    :param df: dataframe
    :param path_: local file path
    """
    if LOG:
        print(f"Saving {path_}")
    df.to_csv(path_, index=False)


def load_dataframe(path_: str) -> pd.DataFrame:
    """
    Load dataframe from disk
    :param path_: local file path
    :return: dataframe
    """
    if LOG:
        print(f"Loading {path_}")
    return pd.read_csv(path_)
