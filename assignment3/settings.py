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

VECTOR_MAPPING = 'info_mapping'
VECTOR_INVERSE_MAPPING = 'info_inverse_mapping'
IDF_PERMUTATION = 'info_idf_permutation'
TERMS_INFO = 'info_terms'
TERMS_INFO_IDF = 'info_terms_idf'

# EVALUATION
EVALUATION_DIR = 'evaluation'
EXACT_SOLUTION = 'exact_solution'

# SCRIPT
SCRIPT_DIR = 'scripts'

# IMAGE DIRECTORY
IMAGES_DIR = "images"

# TOKENIZATION SETTINGS
EXTRA_PUNCTUATION = '•'
DEFAULT_LANGUAGE = 'english'

# LOGGER
IO_LOG = False  # if to print OD operation
LOG = True      # if to print operation
