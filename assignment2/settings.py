"""
Global setting file
"""

import os.path as path

DATASETS_DIR_NAME = "datasets"

IMAGES_DIR_NAME = "images"

TEST = path.join('qrels', 'test.tsv')
CORPUS = 'corpus.jsonl'
QUERIES = 'queries.jsonl'

VECTOR_DIR = 'vectors'
SPARSE_DOCS = 'documents_sparse'
SPARSE_QUERY = 'query_sparse'
DENSE_DOCS = 'documents_dense'
DENSE_QUERY = 'query_dense'

SCORE_DIR = 'scores'
SPARSE_SCORES = 'sparse_scores'
DENSE_SCORES = 'dense_scores'
FULL_SCORES = 'full_scores'

EVALUATION_DIR = "evaluation"
EVALUATION_FILE = "evaluation"

EXTRA_PUNCTUATION = '•'
DEFAULT_LANGUAGE = 'english'

LOG = False
