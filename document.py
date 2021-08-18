
class Document:
    def __init__(self, tweet_id, tokenized_text = None, term_doc_dictionary = None,  doc_length = 0, max_tf = 0):
        """
        :param tweet_id: tweet id
        :param tokenized_text: tokenized_text
        :param term_doc_dictionary: dictionary of term and documents.
        :param doc_length: doc length
        :param max_tf: max_tf of the document
        """
        self.tweet_id = tweet_id
        self.term_doc_dictionary = term_doc_dictionary
        self.tokenized_text = tokenized_text
        self.doc_length = doc_length
        self.max_tf = max_tf
