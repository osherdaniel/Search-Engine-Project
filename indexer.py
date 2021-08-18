import math
import os
import pickle

from numpy import long


class Indexer:

    def __init__(self, config):
        self.inverted_idx = {}
        self.config = config

        if self.config.toStem is True:
            self.out = self.config.saveFilesWithStem
        else:
            self.out = self.config.saveFilesWithoutStem
        self.out += '\\'

        self.A_dictionary = {}
        self.B_dictionary = {}
        self.C_dictionary = {}
        self.D_dictionary = {}
        self.E_dictionary = {}
        self.F_dictionary = {}
        self.G_dictionary = {}
        self.H_dictionary = {}
        self.I_dictionary = {}
        self.J_dictionary = {}
        self.K_dictionary = {}
        self.L_dictionary = {}
        self.M_dictionary = {}
        self.N_dictionary = {}
        self.O_dictionary = {}
        self.P_dictionary = {}
        self.N_dictionary = {}
        self.O_dictionary = {}
        self.P_dictionary = {}
        self.N_dictionary = {}
        self.O_dictionary = {}
        self.P_dictionary = {}
        self.Q_dictionary = {}
        self.R_dictionary = {}
        self.S_dictionary = {}
        self.T_dictionary = {}
        self.U_dictionary = {}
        self.V_dictionary = {}
        self.W_dictionary = {}
        self.X_dictionary = {}
        self.Y_dictionary = {}
        self.Z_dictionary = {}
        self.hashTag_dictionary = {}
        self.tag_dictionary = {}
        self.numbers_dictionary = {}
        self.tweet_info = {}

        self.upper_letters_word = []
        self.bow_corpus = []

        self.ABC_dictionary = {'a': self.A_dictionary, 'b': self.B_dictionary, 'c': self.C_dictionary,
                               'd': self.D_dictionary,
                               'e': self.E_dictionary, 'f': self.F_dictionary, 'g': self.G_dictionary,
                               'h': self.H_dictionary,
                               'i': self.I_dictionary, 'j': self.J_dictionary, 'k': self.K_dictionary,
                               'l': self.L_dictionary,
                               'm': self.M_dictionary, 'n': self.N_dictionary, 'o': self.O_dictionary,
                               'p': self.P_dictionary,
                               'q': self.Q_dictionary, 'r': self.R_dictionary, 's': self.S_dictionary,
                               't': self.T_dictionary,
                               'u': self.U_dictionary, 'v': self.V_dictionary, 'w': self.W_dictionary,
                               'x': self.X_dictionary,
                               'y': self.Y_dictionary, 'z': self.Z_dictionary, 'numbers': self.numbers_dictionary,
                               'hashTag': self.hashTag_dictionary, 'tag': self.tag_dictionary,
                               'tweetInfo': self.tweet_info, 'bowCorpus': self.bow_corpus}

        self.docCounter = 0
        self.dicCounter = 1
        self.lda_indexer = 0

    def clear_all_dic(self):
        """
        Clear all dictionary after write to disk.
        :return: -
        """
        for key in self.ABC_dictionary:
            if key != 'bowCorpus':
                self.ABC_dictionary.get(key).clear()
            else:
                self.bow_corpus = []

    def add_new_doc(self, document):
        """
        This function perform indexing process for a document object.
        Saved information is captures via two dictionaries ('inverted index' and 'posting')
        :param document: a document need to be indexed.
        :return: -
        """
        self.docCounter += 1

        document_dictionary = document.term_doc_dictionary

        for term in document_dictionary:
            try:
                letter = term[0].lower()
                if letter == '#':
                    postingDict = self.hashTag_dictionary
                elif letter == '@':
                    postingDict = self.tag_dictionary
                elif letter.isdigit():
                    postingDict = self.numbers_dictionary
                elif letter in self.ABC_dictionary:
                    postingDict = self.ABC_dictionary.get(letter)
                else:
                    continue

                if term not in self.inverted_idx:
                    self.inverted_idx[term] = [1, document_dictionary[term]]
                    postingDict[term] = []

                    if term[0].isupper():
                        self.upper_letters_word.append(term)
                else:
                    if term in postingDict:
                        self.inverted_idx[term][0] += 1
                        self.inverted_idx[term][1] += document_dictionary[term]
                    else:
                        postingDict[term] = []
                if document.max_tf != 0:
                    tf = document_dictionary[term] / document.max_tf
                else:
                    tf = 0
                postingDict[term].append((document.tweet_id, document_dictionary[term], tf))

            except:
                print('problem with the following key {}'.format(term[0]), term)

        self.tweet_info[document.tweet_id] = [self.lda_indexer, document.max_tf, len(document.term_doc_dictionary)]

        self.lda_indexer += 1

        self.bow_corpus.append(document.tokenized_text)
        #self.bow_corpus.append(LDA.dictionary.doc2bow(document.tokenized_text, allow_update = True))

        if self.docCounter > 575000:
            self.write_posting_to_disk()
            self.clear_all_dic()
            self.docCounter = 0

    def write_posting_to_disk(self):
        """
        Writing all dictionary to disk.
        :return: -
        """
        for key in self.ABC_dictionary:
            if len(self.ABC_dictionary.get(key)) > 0:
                for word in self.ABC_dictionary[key]:
                    if key != 'tweetInfo' and key != 'bowCorpus':
                        self.ABC_dictionary[key][word].sort()

                file_name = self.out + str(key) + str(self.dicCounter) + "_posting_file.pickle"
                outfile = open(file_name, 'wb')
                pickle.dump(self.ABC_dictionary.get(key), outfile)
                outfile.close()

        self.dicCounter += 1

    def merge_files(self):
        """
        Merging all posting files.
        This function also calculate IDF to each term, and handling upper and lower letters rule.
        :return: -
        """
        self.upper_letters_word.sort()
        tweet_tfidf = {}

        index_upper = 0
        for letter in self.ABC_dictionary:
            file_counter = 1

            file_name_1 = self.out + str(letter) + str(file_counter) + "_posting_file.pickle"
            file_counter += 1

            if os.path.exists(file_name_1):
                with open(file_name_1, "rb") as f_1:
                    data_1 = pickle.load(f_1)

                while file_counter < self.dicCounter:
                    file_name_2 = self.out + str(letter) + str(file_counter) + "_posting_file.pickle"
                    if os.path.exists(file_name_2):
                        with open(file_name_2, "rb") as f_2:
                            data_2 = pickle.load(f_2)

                        if letter == 'bowCorpus':
                            data_1.extend(data_2)
                        else:
                            for key in data_2:
                                if key in data_1:
                                    data_1[key].extend(data_2[key])
                                else:
                                    data_1[key] = data_2[key]

                        os.remove(file_name_2)
                        file_counter += 1

                index_upper = self.upper_letters(data_1, letter, index_upper)

                if letter != 'tweetInfo' and letter != 'bowCorpus':
                    for key in data_1:

                        # Calculate IDF for each term
                        N = self.lda_indexer
                        idf = math.log2(N / self.inverted_idx[key][0])
                        self.inverted_idx[key]= [self.inverted_idx[key][0], self.inverted_idx[key][1], idf]

                        # Calculate TF-IDF for each term in document
                        for tweetInfo in data_1[key]:
                            tweet_id = tweetInfo[0]
                            tfidf = tweetInfo[2] * idf
                            if tweet_id in tweet_tfidf:
                                tweet_tfidf[tweet_id] += math.pow(tfidf, 2)
                            else:
                                tweet_tfidf[tweet_id] = math.pow(tfidf, 2)

                        data_1[key].sort()

                        index = 0
                        while index < len(data_1[key]):
                            if index != 0:
                                t_1 = str(long(data_1[key][index][0]) - ID)
                                t_2 = str(data_1[key][index][1])
                                t_3 = str(data_1[key][index][2])
                                t = (t_1, t_2, t_3)
                                ID = long(data_1[key][index][0])
                                data_1[key][index] = t
                            else:
                                ID = long(data_1[key][index][0])
                            index += 1

                os.remove(file_name_1)
                file = self.out + str(letter) + "_posting_file.pickle"
                outfile = open(file, 'wb')
                pickle.dump(data_1, outfile)
                outfile.close()

        file = self.out + "tweet_tfidf_posting_file.pickle"
        outfile = open(file, 'wb')
        pickle.dump(tweet_tfidf, outfile)
        outfile.close()

    def upper_letters(self, data, prev_letter, index):
        """
        This function handling upper and lower letters rule.
        :param data: The posting file that we are working on.
        :param prev_letter: The letter we are working on.
        :param index: The index of the array "upper_letters_word"
        :return: The last index we worked on.
        """
        i = 0
        for i in range(index, len(self.upper_letters_word)):
            term = self.upper_letters_word[i]
            letter = term[0].lower()
            if letter == prev_letter:
                if term.lower() in data:
                    data[term.lower()].extend(data[term])
                    del data[term]

                    value_0 = self.inverted_idx[term][0] + self.inverted_idx[term.lower()][0]
                    value_1 = self.inverted_idx[term][1] + self.inverted_idx[term.lower()][1]
                    self.inverted_idx[term.lower()] = [value_0, value_1]
                    del self.inverted_idx[term]

                elif not term.upper() in data:
                    data[term.upper()] = data[term]
                    del data[term]

                    value_0 = self.inverted_idx[term][0]
                    value_1 = self.inverted_idx[term][1]
                    self.inverted_idx[term.upper()] = [value_0, value_1]
                    del self.inverted_idx[term]
            else:
                break
        return i

    def names_and_entity(self, name_or_entity):
        """
        This function handling name and entity rule.
        :param name_or_entity: List of all names and entity we found in the parser.
        :return:
        """
        for term in name_or_entity:
            if len(name_or_entity[term]) == 1:
                [(k, v)] = name_or_entity[term].items()

            if len(name_or_entity[term]) > 1 or v[0] > 1:
                s = 0
                for tweet in name_or_entity[term]:
                    s += int(name_or_entity[term][tweet][0])
                self.inverted_idx[term] = [s, len(name_or_entity[term])]

                letter = term[0].lower()

                postingDict = self.ABC_dictionary.get(letter)
                if postingDict is not None:
                    if term not in postingDict:
                        postingDict[term] = []
                    for tweet_id in name_or_entity[term]:
                        tf = name_or_entity[term][tweet_id][0] / name_or_entity[term][tweet_id][1]
                        postingDict[term].append((tweet_id, name_or_entity[term][tweet_id], tf))

        self.docCounter += 1
