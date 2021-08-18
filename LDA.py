import datetime
import itertools
import os
import pickle

import gensim

import utils
from MyCorpus import MyCorpus
from gensim import corpora


def create_parmaters():
    """
    Create LDA data.
    :return:
    """
    file_name = LDA.out + "bowCorpus_posting_file.pickle"
    with open(file_name, "rb") as f:
        LDA.data = pickle.load(f)


class LDA(object):
    dictionary = gensim.corpora.Dictionary()
    lda_model = None
    topic_dict = {i: [] for i in range(10)}
    data = None
    out = None
    doc_dict = {}
    tweet_info_data = None

    @staticmethod
    def create_model(out, stemming):
        """
         Create the LDA model.
        :param out: The path of the posting files.
        :param stemming: Boolean param.
        :return:
        """
        LDA.out = out
        create_parmaters()

        LDA.lda_model = gensim.models.LdaMulticore(MyCorpus(LDA.dictionary, LDA.data), num_topics=10,
                                                   id2word=LDA.dictionary, passes=1, minimum_probability=0.0, workers=5)
        if stemming:
            LDA.lda_model.save("ModelWIthStem\\LDA_Model")
        else:
            LDA.lda_model.save("ModelWithoutStem\\LDA_Model")

    def grouper(iterable, N, fillvalue=None):
        args = [iter(iterable)] * N
        return itertools.zip_longest(*args, fillvalue=fillvalue)

    @staticmethod
    def load_model(out, stemming):
        """
        Load the model from disk and create Documents-Topics matrix.
        :param out: The path of the posting files.
        :param stemming: Boolean param.
        :return: -
        """
        LDA.out = out

        if stemming:
            LDA.lda_model = gensim.models.LdaModel.load("ModelWIthStem\\LDA_Model")
        else:
            LDA.lda_model = gensim.models.LdaModel.load("ModelWithoutStem\\LDA_Model")

        LDA.dictionary = LDA.lda_model.id2word

        file_name = LDA.out + "bowCorpus_posting_file.pickle"
        with open(file_name, "rb") as f:
            LDA.data = pickle.load(f)

        file_name = LDA.out + str("tweetInfo_posting_file.pickle")
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                LDA.tweet_info_data = pickle.load(f)

        file_name = LDA.out + str("tweet_tfidf_posting_file.pickle")
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                LDA.tweets_tfidf = pickle.load(f)


    @staticmethod
    def batch_matrix(similarity):
        """
        Creating Documents-Topics matrix.
        :return: -
        """

        for key in similarity:
            if key not in LDA.doc_dict:
                docID = LDA.tweet_info_data[key][0]
                doc_bow = LDA.dictionary.doc2bow(LDA.data[docID])
                LDA.doc_dict[key] = LDA.lda_model.get_document_topics(bow=doc_bow, minimum_probability=0.0)

