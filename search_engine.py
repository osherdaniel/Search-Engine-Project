import datetime

from reader import ReadFile
from configuration import ConfigClass
from parser_module import Parse
from indexer import Indexer
from searcher import Searcher
import utils
from LDA import LDA


def run_engine(corpus_path, output_path, stemming):
    number_of_documents = 0

    config = ConfigClass(corpus_path, output_path, stemming)
    r = ReadFile(corpus_path=config.get_corpusPath())
    p = Parse(config.toStem)
    indexer = Indexer(config)

    #documents_list = r.read_file(file_name='Test3.parquet')
    documents_list = r.read_files()

    for idx, document in enumerate(documents_list):
        parsed_document = p.parse_doc(document)
        number_of_documents += 1
        indexer.add_new_doc(parsed_document)

    indexer.names_and_entity(p.names_and_entity_dictionary)

    if indexer.docCounter > 0:
        indexer.write_posting_to_disk()

    indexer.merge_files()

    #LDA.create_model(indexer.out, stemming)
    LDA.load_model(indexer.out, stemming)
    utils.save_obj(indexer.inverted_idx, "inverted_idx")


def load_index():
    inverted_index = utils.load_obj("inverted_idx")
    return inverted_index


def search_and_rank_query(query, inverted_index, k, stemming, output_path):
    if stemming is True:
        out = output_path + "\\WithStem"
    else:
        out = output_path + "\\WithoutStem"
    out += '\\'

    p = Parse(stemming)
    query_as_list = p.parse_sentence(query)
    searcher = Searcher(inverted_index, stemming, out)
    relevant_docs = searcher.relevant_docs_from_posting(query_as_list)
    ranked_docs = searcher.ranker.rank_relevant_doc(relevant_docs)
    return searcher.ranker.retrieve_top_k(ranked_docs, k)


def main(corpus_path, output_path, stemming, queries, num_docs_to_retrieve):
    run_engine(corpus_path, output_path, stemming)
    inverted_index = load_index()

    if not isinstance(queries, list):
        with open(queries, encoding = "utf8") as file:
            queries = file.readlines()

    for query in queries:
        query = query.rstrip()
        if query is not None:
            for doc_tuple in search_and_rank_query(query, inverted_index, num_docs_to_retrieve, stemming, output_path):
                print('Tweet id: {} Score: {}'.format(doc_tuple[0], doc_tuple[1]))


# def remove_all_files():
#     for f in os.listdir("C:\Project\Search_Engine-master\Output Files\WithoutStem"):
#         if not f.endswith(".pickle"):
#             continue
#         os.remove(os.path.join("C:\Project\Search_Engine-master\Output Files\WithoutStem", f))
