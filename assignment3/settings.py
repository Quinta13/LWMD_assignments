"""
This file contain configuration settings
"""

import os.path as path

# INPUT FILES and DIRECTORIES
DATASETS_DIR_NAME = "datasets"
CORPUS = 'corpus.jsonl'
QUERIES = 'queries.jsonl'
TEST = path.join('qrels', 'test.tsv')


# VECTORIZATION FILES and DIRECTORIES
VECTOR_DIR = 'vector'
VECTOR_FILE = 'vector'

IDF_PERMUTATION = 'info_idf_permutation'
VECTOR_MAPPING = 'info_mapping'
VECTOR_INVERSE_MAPPING = 'info_inverse_mapping'

# EVALUATION
EVALUATION_DIR = 'evaluation'
EXACT_SOLUTION = 'exact_solution'

# IMAGE DIRECTORY
IMAGES_DIR = "images"

# TOKENIZATION SETTINGS
EXTRA_PUNCTUATION = '•'
DEFAULT_LANGUAGE = 'english'

# LOGGER
IO_LOG = False  # if to print OD operation
LOG = True  # if to print operation
