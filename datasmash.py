__author__ = 'pslii'
import numpy as np
import matplotlib.pyplot as plt

def entropy_quantization(data, n_states):
    """
    Generates quantization based on equal entropy scheme.
    :param data:
    :param n_states:
    :return:
    """
    percentages = np.linspace(0, 100, n_states + 1)
    splits = np.percentile(data, list(percentages))

    quantize = np.empty_like(data)
    for i in range(n_states):
        subset = np.where((data >= splits[i]) & (data <= splits[i + 1]))[0]
        quantize[subset] = i
    return quantize

class DataSmash:
    """
    Data Smashing algorithm
    Chattopadhyay & Lipson 2014

    http://rsif.royalsocietypublishing.org/content/suppl/2014/09/30/rsif.2014.0826.DC1/rsif20140826supp1.pdf
    http://rsif.royalsocietypublishing.org/content/11/101/20140826.short?rss=1
    """

    def __init__(self, n_states, l_max=3):
        self.n_states = n_states
        self.alphabet = set(np.arange(n_states, dtype=int))
        self.powersets = self._powerset(self.alphabet, l_max)
        self.regex_functions = self._regex(self.powersets)

    def _regex(self, powersets):
        """
        Pre-compiles regex strings

        :param powersets: List of lists output by _powerset
        :return: List of lists containing compiled regex functions
        """
        import re
        regex_functions = []
        for powerset in powersets:
            regex_strings = ['(?='+set_string+')' for set_string in powerset]
            regex_functions_l = [re.compile(regex_string) for regex_string in regex_strings]
            regex_functions.append(regex_functions_l)
        return regex_functions

    @staticmethod
    def _powerset(self, alphabet, l_max):
        """
        Generates powerset of length 1 or greater from the given alphabet.
        :param l_max: Maximum length substring to return. Note: algorithm is super-exponential in l_max.
        :return: A list of lists
        """
        assert l_max >=1, "Error: L_max must be greater than or equal to 1."

        import itertools as it
        powerset = []
        for l in range(1, l_max+1):
            l_subset = []
            for x in it.product(alphabet, repeat=l):
                l_subset.append("".join(map(str, x)))
            powerset.append(l_subset)
        return powerset

    def stream_copy(self, s):
        """
        Generates an independent sample path from the same hidden stochastic source

        :param s: quantized data stream
        :return: independent stream copy
        """
        length = len(s)
        fwn = np.random.random_integers(0, self.n_states - 1, length)
        stream_sum = self.stream_summation(s, fwn)

        # Prevent from returning completely empty stream
        if len(stream_sum) == 0:
            return self.stream_copy(s)
        else:
            return stream_sum

    def stream_summation(self, s1, s2):
        """
        Performs a stream summation over two streams
        :param s1:
        :param s2:
        :return:
        """
        min_length = min(len(s1), len(s2))
        match = np.where(s1[:min_length] == s2[:min_length])[0]
        return s1[match]

    def stream_inversion(self, s, verbose=False):
        s_copies = []
        s_lengths = np.empty(self.n_states - 1, dtype=int)
        # generate \Sigma-1 independent copies of s
        for i in range(self.n_states - 1):
            s_copy = self.stream_copy(s)
            s_lengths[i] = len(s_copy)
            s_copies.append(s_copy)
            if verbose:
                print s_copy

        # truncate arrays
        min_len = min(s_lengths)
        s_copyarr = np.empty((self.n_states - 1, min_len))
        for i, s_copy in enumerate(s_copies):
            s_copyarr[i, :] = s_copy[:min_len]

        # read current symbols \sigma_i from s_i
        s_invert = []
        for i in range(min_len):
            int_set = set(s_copyarr[:, i])
            if len(int_set) == self.n_states - 1:
                s_invert.append((self.alphabet - int_set).pop())
        if verbose:
            print s_invert
        return np.array(s_invert)

    def phi(self, s_in, x):
        """
        Symbolic derivative
        supplement, eqn 13
        :param s: integer string, list, or array
        :param x: substring to be identified in array
        :return:
        """
        occurrences = np.zeros(self.n_states, dtype=float)
        if len(s_in) == 0:
            return occurrences

        import re
        s = ''.join(s_in) if not isinstance(s_in, str) else s_in
        for i, letter in enumerate(self.alphabet):
            regex = '(?=' + x + str(letter) + ')'
            occurrences[i] = len(re.findall(regex, s))
        if occurrences.sum() == 0:
            return occurrences
        else:
            return occurrences / occurrences.sum()

    def fwn_deviation(self, s, l_max):
        if len(s) == 0:
            return -1.0

        import itertools as it

        n_states = float(self.n_states)
        s_in = "".join(map(str, s))
        uniform_vector = np.ones(n_states) / n_states
        sum = 0.0
        for l in range(l_max):
            denominator = n_states ** (2 * l)
            for x in it.product(self.alphabet, repeat=l):
                substring = "".join(map(str, x))
                numerator = np.max(np.abs(self.phi(s_in, substring) - uniform_vector))
                sum += (numerator / denominator)
        return sum * (n_states - 1) / n_states

    def annihilation_circuit(self, s1, s2, l_max=3):
        """
        Annihilation circuit
        :param s1:
        :param s2:
        :return:
        """
        s1_inv, s2_inv = self.stream_inversion(s1), self.stream_inversion(s2)
        ssum11 = self.stream_summation(s1, s1_inv)
        ssum12 = self.stream_summation(s1_inv, s2)
        ssum21 = self.stream_summation(s1, s2_inv)
        ssum22 = self.stream_summation(s2, s2_inv)

        eps11 = self.fwn_deviation(ssum11, l_max)
        eps12 = self.fwn_deviation(ssum12, l_max)
        eps21 = self.fwn_deviation(ssum21, l_max)
        eps22 = self.fwn_deviation(ssum22, l_max)
        return eps11, eps12, eps21, eps22