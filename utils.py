#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np


def parse_text(text: str, sep: str = '') -> (list[str], list[str]):
    """Parses some text and generates a list of unique words and the initial text as a list of words

    :param text: String to parse
    :type text: str
    :param sep: Separator character to split string with, defaults to ''
    :type sep: str, optional
    :return: A tuple with two lists of strings, first one being the vocabulary and the second being the split input text
    :rtype: tuple[list[str], list[str]]
    """
    if not sep:
        data = list(text)
        vocab = list(set(data))
    else:
        data = text.lower().split(sep)
        vocab = list(set(data))
    vocab.sort()
    return vocab, data


def one_hot(x: list, vocab: list) -> np.ndarray:
    """Transforms the inputted list of strings to a one-hot embeddings array using the specified vocabulary.

    :param x: The list of strings to transform.
    :type x: list[str]
    :param vocab: The list of unique words to use for the word indices.
    :type vocab: list[str]
    :return: A matrix with binary values, where each row contains 1 in the index of the word specified by the column.
    :rtype: np.ndarray
    """
    one_hot_x = np.zeros((len(x), len(vocab)))
    for idx in range(len(x)):
        one_idx = vocab.index(x[idx])
        one_hot_x[idx, one_idx] = 1
    return one_hot_x


def word2idx(x: list, vocab: list) -> np.ndarray:
    """Transforms a list of strings to a vector of indices based on the given vocabulary.

    :param x: The list of strings to transform.
    :type x: list[str]
    :param vocab: The vocabulary to get the indices from.
    :type vocab: list[str]
    :return: A vector filled with each word's index in the vocabulary list.
    :rtype: np.ndarray
    """
    idx = np.zeros((1, len(x)), dtype=np.int32)
    for ind in range(len(x)):
        idx[0, ind] = vocab.index(x[ind])
    return idx


def vec2word(x: np.ndarray, vocab: list) -> str:
    """Transforms a vector of indices to a sentence based on the given vocabulary.

    :param x: A vector of indices.
    :type x: np.ndarray
    :param vocab: The vocabulary to associate the indices with words.
    :type vocab: list[str]
    :return: The concatenation of the words specified by the indices in x.
    :rtype: str
    """
    words = ''
    for idx in range(len(x)):
        words += vocab[x[idx]]
    return words.strip()
