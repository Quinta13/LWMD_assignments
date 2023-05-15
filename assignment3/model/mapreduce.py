"""
This module ... TODO
"""
from typing import List, Tuple


def document_map(doc_info: Tuple[str, List[Tuple[int, float]]]) -> List[Tuple[int, Tuple[str, List[Tuple[int, float]]]]]:
    """
    Mapping function
    :param doc_info: document information as a pair (doc-id ; list(term-id, value)),
        where the second value of the pair represent entries of a csr vector
    :return: list of key-value pair (term-id : (doc-id ; list(term-id, value))) for each term appearing in the document
    """

    doc_id, sparse_entries = doc_info

    return [
        (term_id, (doc_id, sparse_entries))
        for term_id, value in sparse_entries
    ]