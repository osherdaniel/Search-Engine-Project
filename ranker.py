import os
import pickle


class Ranker:
    def __init__(self):
        pass

    @staticmethod
    def rank_relevant_doc(relevant_doc):
        """
        This function provides rank for each relevant document and sorts them by their scores.
        The current score considers solely the number of terms shared by the tweet (full_text) and query.
        :param relevant_doc: dictionary of documents that contains at least one term from the query.
        :return: sorted list of documents by score
        """
        relevant_doc = sorted(relevant_doc.items(), key=lambda item: item[1], reverse=True)

        if len(relevant_doc) > 2000:
            relevant_doc = relevant_doc[:2000]

        return relevant_doc

    @staticmethod
    def retrieve_top_k(sorted_relevant_doc, k=1):
        """
        return a list of top K tweets based on their ranking from highest to lowest
        :param sorted_relevant_doc: list of all candidates docs.
        :param k: Number of top document to return
        :return: list of relevant document
        """
        if len(sorted_relevant_doc) > k:
            sorted_relevant_doc = sorted_relevant_doc[:k]
        return sorted_relevant_doc
