import collections
from functools import reduce
import operator

import pandas as pd


class Document(object):
    """Class to define document cleaning and parsing"""

    def __init__(self, raw_data: str):
        self.raw_data = raw_data
        self.clean_doc = self.clean_document(raw_data)
        self.word_counts = self.parse_document(self.clean_doc)

    @staticmethod
    def clean_document(raw_doc: str):
        """Clean a raw document

        Tips for improvement:
        * Special character/pattern cleaning (e.g. email, phone number, ...)
        * Spell Checker? Stemming? (e.g. Hunspell, Porter, pyspellchecker)
        """
        return raw_doc.lower()

    @staticmethod
    def parse_document(clean_doc: str):
        """Parse document to generate a word count"""
        return collections.Counter(clean_doc.split(" "))


class NaiveBayes(object):
    """Naive Bayes Classifier"""

    def __init__(self, word_freq=None, word_probs=None, min_prob=1e-5):
        """
        word_freq or word_probs must be specified

        PARAMETERS
        word_freq (dict or pd.DataFrame): optional; word frequency
            * index = words
                - "__total__" means the actual total number of words if data is only a subset
            * columns = ["spam", "ham"]
            * values = int representing word frequency
        word_probs (dict or pd.DataFrame): optional; word probabilities
            * index = words
            * columns = ["spam", "ham"]
            * values = float representing word ratio (i.e. word probability)

        min_prob (float): optional, defaults to 1e-5, minimum probability (to handle 0 probabilities)
        """
        assert word_freq or word_probs, "either word_freq or word_probs must be specified"

        if word_probs:
            self.p_df = word_probs.copy() if isinstance(
                word_probs, pd.DataFrame) else pd.DataFrame(word_probs)
        else:
            self.wf_df = word_freq.copy() if isinstance(
                word_freq, pd.DataFrame) else pd.DataFrame(word_freq)

            col_order = ["spam", "ham"]
            if "__total__" in self.wf_df.index:
                assert self.wf_df.loc["__total__", :].notnull(
                ).all(), "both totals must be specified"
                self.p_df = (
                    self.wf_df[col_order].fillna(0)
                    / self.wf_df.loc["__total__", col_order]
                )

            else:
                self.p_df = self.wf_df[col_order].transform(
                    lambda x: x / x.sum()).fillna(0)

        self.p_df = self.p_df.applymap(lambda x: max(x, min_prob))
        self.p_df["likelihood_ratio"] = self.p_df["spam"] / self.p_df["ham"]

    def posterior_spam(self, document: str, prior_odds: float=1, prior_p: float=None, freq_factor: bool=True):
        """Calculate posterior odds of a document being spam

        PARAMETERS
        document (str): raw document to parse
        prior_odds (float): optional, defaults to `1`; prior odds of spam messages
        prior_p (float): optional; to provide prior as a probability instead of odds
        freq_factor (bool): optional, defaults to `True`; flag whether to consider word frequency in `document`
            * If `False`, "million million million" will be the same as "million"
        """
        prior = prior_odds or (prior_p) / (1 - prior_p)

        assert prior > 0, "prior_odds must be > 0, or prior_p must be between 0 and 1"

        word_counts = document.word_counts if isinstance(
            document, Document) else Document(document).word_counts

        factors = [
            (self.p_df.loc[word, "likelihood_ratio"]
             if word in self.p_df.index else 1) ** (count if freq_factor else 1)
            for word, count in word_counts.items() if count > 0
        ]

        return prior * reduce(operator.mul, factors)


def answer_exercises():
    # word_freq can be generated from "spam" and "ham" raw data using the Document class
    word_freq = {
        'spam': {
            'million': 156, 'dollars': 29, 'adclick': 51, 'conferences': 0, '__total__': 95791
        },
        'ham': {
            'million': 98, 'dollars': 119, 'adclick': 0, 'conferences': 12, '__total__': 306438
        }
    }

    clf = NaiveBayes(word_freq)
    # modification for exercise due to precision limitation
    clf.p_df["likelihood_ratio"] = clf.p_df["likelihood_ratio"].apply(
        lambda x: round(x, 1))
    print(clf.p_df)
    print(clf.posterior_spam("million"))
    print(clf.posterior_spam("million dollars adclick conferences"))


if __name__ == "__main__":
    answer_exercises()
