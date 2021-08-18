import math
import os
import pickle

from gensim.matutils import cossim
from parser_module import Parse
from ranker import Ranker
from LDA import LDA
from numpy import long


class Searcher:

    def __init__(self, inverted_index=None, stemming=None, out=None):
        """
        :param inverted_index: dictionary of inverted index
        """
        self.parser = Parse(stemming)
        self.ranker = Ranker()
        self.inverted_index = inverted_index

        self.out = out

    def relevant_docs_from_posting(self, query):
        """
        This function loads the posting list and count the amount of relevant documents per term.
        :param out:
        :param query: query
        :return: dictionary of relevant documents.
        """
        if len(query) == 0:
            return {}

        bow_query = LDA.dictionary.doc2bow(query)

        # --- TF-IDF ---
        data = {}
        similarity = {}

        # --- Upper and lower letters ---
        for index in range(0, len(query)):
            word = query[index]
            if word.lower() in self.inverted_index:
                query[index] = word.lower()
            elif word.upper() in self.inverted_index:
                query[index] = word.upper()
            else:
                continue

        # --- Name or entity ---
        for index in range(0, len(query)):
            word = query[index]
            if len(word) > 1 and word[0].isupper():
                j = index + 1
                name_or_entity = word
                while j < len(query):
                    if len(query[j]) > 1 and query[j][0].isupper():
                        name_or_entity += " " + query[j]
                        j += 1
                    else:
                        break
                query.append(name_or_entity)

        query.sort()
        prev_letter = query[0][0].lower()

        if prev_letter == '#':
            prev_letter = 'hashTag'
        elif prev_letter == '@':
            prev_letter = 'tag'
        elif prev_letter.isdigit():
            prev_letter = 'numbers'

        file_name = self.out + str(prev_letter) + "_posting_file.pickle"
        if os.path.exists(file_name):
            with open(file_name, "rb") as f:
                data = pickle.load(f)

        for term in query:
            letter = term[0].lower()

            if letter == '#':
                letter = 'hashTag'
            elif letter == '@':
                letter = 'tag'
            elif letter.isdigit():
                letter = 'numbers'

            if letter != prev_letter:
                file_name = self.out + str(letter) + "_posting_file.pickle"
                if os.path.exists(file_name):
                    with open(file_name, "rb") as f_2:
                        data = pickle.load(f_2)

            if term in data:
                data_of_term = data[term]
                addition_tweet_id = 0
                for tweet in data_of_term:
                    tweet_id = long(tweet[0]) + addition_tweet_id
                    addition_tweet_id = tweet_id
                    tweet_id = str(tweet_id)

                    tf = float(tweet[2])
                    idf = self.inverted_index[term][2]
                    tf_idf = tf * idf

                    if tweet_id in similarity:
                        similarity[tweet_id] += tf_idf
                    else:
                        similarity[tweet_id] = tf_idf

                prev_letter = letter

        similarity = sorted(similarity.items(), key=lambda item: item[1], reverse=True)
        if len(similarity) > 4000:
            similarity = dict(similarity[:4000])
        else:
            similarity = dict(similarity)

        # --- Vector Similarity ---
        vector_query = LDA.lda_model.get_document_topics(bow=bow_query, minimum_probability=0.0)

        # -- Compound Ranking --
        ranker = {}
        w_query = len(query)

        LDA.batch_matrix(similarity)

        for tweet_id in similarity:
            if tweet_id in LDA.tweets_tfidf:
                w_doc = LDA.tweets_tfidf[tweet_id]
                cos_sim_tfidf = similarity[tweet_id] / math.sqrt((w_doc * w_query))

                vector_doc = LDA.doc_dict[tweet_id]
                cos_sim_vectors = cossim(vector_query, vector_doc)
                ranker[tweet_id] = 0.6 * cos_sim_vectors + 0.4 * cos_sim_tfidf

        return ranker
