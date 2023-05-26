"""
This script is used to launch SparkApplication
"""

import sys
import logging
import os
import time

from pyspark.sql import SparkSession

sys.path.append('/home/sebaq/Documents/GitHub/LWMD_assignments')

from assignment3.io_ import get_script_dir
from assignment3.model.documents import DocumentVectors

LOG_NAME = 'logfile_trial.log'
APP_NAME = "PairswiseSimilarity"

DEFAULT_DATA = 'small'
DEFAULT_SIM = 0.9
DEFAULT_IDF = True

# Configure logging
logging.basicConfig(filename=os.path.join(get_script_dir(), LOG_NAME),
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def document_map(
        doc_info: tuple[str, int, list[tuple[int, float]]]
) -> list[tuple[str, tuple[str, int, list[tuple[int, float]]]]]:
    """
    Mapping function
    :param doc_info: document information is represented as a triple:
        - doc-id, represented as a string
        - term-threshold, referring to the index of a specific column up to which do not map terms
        - document vector, a sparse vector as a list of pairs (column, value) for each non-zero entries,
            where the column is actually a term-id
    :return: list of key-value pairs:
        - key: term-id, which is actually a column index
        - value: consists of a triple:
            - doc-id  (the same as input)
            - term-id (the same as the key)
            - document vector (the same as input)
    """

    # unpacking
    doc_id: str
    term_threshold: int
    sparse_entries: list[tuple[int, float]]
    doc_id, term_threshold, sparse_entries = doc_info

    mapped: list[tuple[str, tuple[str, int, list[tuple[int, float]]]]] = [

        (str(term_id), (doc_id, term_id, sparse_entries))
        for term_id, value in sparse_entries  # document terms by using non-zero entries
        if term_id > term_threshold  # OPTIMIZATION 1:
        # we only map term with higher term-id with respect to the threshold one
        #  (thus, we only consider columns after the threshold one)
    ]

    return mapped


def documents_reduce(docs: list[tuple[int, int, list[tuple[int, float]]]]) -> list[tuple[tuple[int, int], float]]:
    """
    Reduce function
    :param docs: list of triplets:
        - doc-id
        - term-id (actually a column index of the vector)
        - document vector as a sparse matrix of pairs (column, value)
    :return: list of tuples:
        - the first element is the pair of documents represented by their doc-id
        - the second element represent their cosine-similarity
    """

    # list of output pairs
    pairs = []

    # DOC-TERMS HEURISTIC pt. 1 - sort items for document length
    docs = sorted(docs, key=lambda x: len(x[2]), reverse=True)

    # total number of documents
    n_docs = len(docs)

    # loop among all possible pairs
    for i in range(n_docs - 1):

        doc1_id, term_id, doc1 = docs[i]

        for j in range(i + 1, n_docs):

            doc2_id, _, doc2 = docs[j]  # since the operation is an aggregation by key,
            # term_id is expected to be the same

            # DOC-TERMS HEURISTIC pt. 2 - skip if too-high length mismatch
            if len(doc1) / len(doc2) > 1.3:
                break

            # ----------------- OPTIMIZATION 2 -----------------

            # collect term-ids of each document
            terms_1: list[int] = [t_id1 for t_id1, _ in doc1]  # term-ids for the first document
            terms_2: list[int] = [t_id2 for t_id2, _ in doc2]  # term-ids for the second document

            # perform their intersection
            common_terms: set[int] = set(terms_1).intersection(terms_2)

            # get the maximum term-id
            max_term: int = max(common_terms)

            # if the maximum term-id is not the same of aggregation key, skip similarity computation
            if term_id != max_term:
                pass

            # --------------------------------------------------

            # Computing similarity with dot-product

            # getting iterator
            iter_doc1 = iter(doc1)
            iter_doc2 = iter(doc2)

            # we assume documents with at least on term
            term1, value1 = next(iter_doc1)
            term2, value2 = next(iter_doc2)

            sim = 0.  # total similarity

            # we use iterators to keep a pointer over term-ids of the two vectors
            # if they have the same term-id, we add its contribution to the cumulative sum and we move both pointers over
            # otherwise we move over the one with smallest term-id

            while True:

                try:
                    if term1 == term2:  # they have common term-id; we add its contribution to final similarity
                        sim += value1 * value2
                        term1, value1 = next(iter_doc1)
                        term2, value2 = next(iter_doc2)
                    elif term1 < term2:  # the first one has a smaller term-id
                        term1, value1 = next(iter_doc1)
                    else:  # the second one has a smaller term-id
                        term2, value2 = next(iter_doc2)
                except StopIteration:  # we scanned all terms of one of the vectors so there's no more term in common
                    break

            # we add the pairwise similarity to final output
            pairs.append(((doc1_id, doc2_id), sim))

    return pairs


if __name__ == "__main__":
    """
    Usage: pairwise_similarity [data_name] [similarity] [idf_order]
    """

    # creating spark session
    spark = SparkSession \
        .builder \
        .appName(APP_NAME) \
        .getOrCreate()

    #
    data_name = str(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
    similarity = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SIM
    idf_order = (int(sys.argv[3]) > 0) if len(sys.argv) > 3 else DEFAULT_IDF

    # Getting input data
    docs_vet = DocumentVectors(data_name=data_name, idf_order=idf_order)
    docs_info = docs_vet.get_documents_info(similarity=similarity)

    rdd = spark.sparkContext.parallelize(docs_info)

    out = (
        rdd.
        # perform mapping
        flatMap(document_map)
        # combine by key
        .combineByKey(lambda x: [x], lambda x, y: x + [y], lambda x, y: x + y)
        # perform reduce
        .flatMapValues(documents_reduce)
        # takes only pairs of doc-ids
        .filter(lambda x: x[1][1] > similarity)
        # get only pairs of documents
        .map(lambda x: x[1][0])
        # remove duplicates
        .distinct()
    )

    t1 = time.perf_counter()

    collected = out.collect()

    t2 = time.perf_counter()

    logging.info(f"Dataset: {data_name} - Threshold: {similarity} - idf col order: {idf_order}")
    logging.info(f"Exec time: {t2 - t1}")
    logging.info(f"Docs {collected}")

    spark.stop()
