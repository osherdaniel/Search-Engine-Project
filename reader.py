import os
import pandas as pd


class ReadFile:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        files_path = [os.path.abspath(x) for x in os.listdir()]

    def read_file(self, file_name):
        """
        This function is reading a parquet file contains several tweets
        The file location is given as a string as an input to this function.
        :param file_name: string - indicates the path to the file we wish to read.
        :return: a dataframe contains tweets.
        """
        full_path = os.path.join(self.corpus_path, file_name)
        df = pd.read_parquet(full_path, engine="pyarrow")
        return df.values.tolist()

    def read_files(self):

        documentList = []
        entries = os.listdir(self.corpus_path)

        entries.remove(".DS_Store")
        for entry in entries:
            sub_entries = os.listdir(os.path.join(self.corpus_path, entry))

            if ".DS_Store" in sub_entries: sub_entries.remove(".DS_Store")
            for sub_entry in sub_entries:
                full_path = os.path.join(entry, sub_entry)
                documentList = documentList + self.read_file(full_path)

        return documentList
