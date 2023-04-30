"""
This file contain configuration settings
"""

import os.path as path

# INPUT FILES and DIRECTORIES
DATASETS_DIR_NAME = "datasets"
TEST = path.join('qrels', 'test.tsv')
CORPUS = 'corpus.jsonl'
QUERIES = 'queries.jsonl'


# VECTORIZATION FILES and DIRECTORIES
VECTOR_DIR = 'vectors'
SPARSE_DOCS = 'documents_sparse'
SPARSE_QUERY = 'query_sparse'
DENSE_DOCS = 'documents_dense'
DENSE_QUERY = 'query_dense'

# SCORES FILES and DIRECTORIES
SCORE_DIR = 'scores'
SPARSE_SCORES = 'sparse_scores'
DENSE_SCORES = 'dense_scores'
FULL_SCORES = 'full_scores'

# EVALUATION FILES and DIRECTORIES
EVALUATION_DIR = "evaluation"
EVALUATION_FILE = "evaluation"

# IMAGE DIRECTORY
IMAGES_DIR_NAME = "images"

# TOKENIZATION SETTINGS
EXTRA_PUNCTUATION = '•'
DEFAULT_LANGUAGE = 'english'

LOG = False  # if to print OS operation
