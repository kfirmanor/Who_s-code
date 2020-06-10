from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import shlex
import re


def data_to_histogram(data, line_func):
    """
    :param data - the text file
    :param line_func - the function you want to use on a line
    :return histogram made from the text file
    """
    text_counter = Counter()
    with open(data) as f:
        for line in f:
            line_counter = Counter(line_func(line))
            text_counter += line_counter
    return text_counter


def word_list(line):
    """"
    :param line - string
    :return a array with all the words inside line
    """
    return re.sub("[^\w]", " ", line).split()


def letter_list(line):
    """"
    :param line - string
    :return a array with all the letters inside line
    """
    return list(line)


def word_histogram(data):
    """"
    :param data - text file
    :return a histogram from all the words inside data
    """
    return data_to_histogram(data, word_list)

def letter_histogram(data):
    """"
    :param data - text file
    :return a histogram from all the letters inside data
    """
    return data_to_histogram(data, letter_list)


def main(counter):
    """"
    for testing
    """
    labels, values = zip(*Counter(counter).items())
    indexes = np.arange(len(labels))
    width = 1
    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)
    plt.show()

if __name__ == "__main__":
    main(data_to_histogram("pytorch_geometric_all_data.txt", word_list))
