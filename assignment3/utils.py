"""
This module contains a collection of utility functions
they have a general purpose and are not strictly related to a specific behaviour of a model class
"""

import string
from typing import List, Collection, Set

from nltk import SnowballStemmer, WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords

from assignment3.settings import EXTRA_PUNCTUATION


def tokenize(text: str, language: str) -> List[str]:
    """
    Tokenize input text performing following operations:
     - removal of punctuation
     - filtering of stop-words, numbers and east-asiatic chars
     - stemming and lemming
    :param text: input text to be parsed
    :param language: language of input text
    :return: list of tokens
    """

    def contains_number(s: str) -> bool:
        """
        :param s: input string
        :return: true if input string contains any number
        """
        return any(char.isdigit() for char in s)

    def contains_east_asia_chars(s):
        """
        :param s: input string
        :return: true if input string contains any east asiatic chars
        """
        return any(0x4E00 <= ord(char) <= 0x9FFF for char in s)

    # Define stemmer and lemmatizer
    stemmer = SnowballStemmer(language)
    lemmatizer = WordNetLemmatizer()

    # Define set of stop words
    stop_words = set(stopwords.words(language))

    # Remove punctuation
    content_: str = text.translate(str.maketrans('', '', string.punctuation + EXTRA_PUNCTUATION))

    # Tokenize text
    tokens: list[str] = word_tokenize(content_)

    # Filter words and apply stemming and lemming
    preprocessed_tokens = []
    for token in tokens:
        token = token.lower()
        if token not in stop_words and \
                not contains_number(token) and \
                not contains_east_asia_chars(token):
            stemmed_token = stemmer.stem(token)
            lemmatized_token = lemmatizer.lemmatize(stemmed_token)
            preprocessed_tokens.append(lemmatized_token)

    return preprocessed_tokens


def jaccard(s1: Set, s2: Set) -> float:
    """
    Return jaccard similarity between two sets in input
    :param s1: first collection
    :param s2: second collection
    :return: jaccard similarity between the two
    """

    if len(s1) == 0 and len(s2) == 0:
        raise Exception("Impossible to compute jaccard similarity between two empty sets")

    intersection = len(s1.intersection(s2))
    union = len(s1.union(s2))

    return intersection / union
