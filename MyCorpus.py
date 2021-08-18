class MyCorpus:

    def __init__(self,dictionary,data):
        self.dictionary = dictionary
        self.data = data

    def __iter__(self):
        for token_list in self.data:
            doc = self.dictionary.doc2bow(token_list)
            yield doc

