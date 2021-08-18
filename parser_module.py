import re

from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
from document import Document

from stemmer import Stemmer


class Parse:

    def __init__(self, to_stem):
        self.stop_words = stopwords.words('english')
        self.all_term_dictionary = {}

        self.names_and_entity_dictionary = {}
        self.doc_name_and_entity = []
        self.to_stem = to_stem

    def parse_sentence(self, text, tweet_id = None, url = None):
        """
        This function tokenize, remove stop words and apply lower case for every word within the text
        :param text:
        :return:
        """
        self.doc_name_and_entity = []

        text = self.remove_emoji(text)

        currency_sign_dictionary = {"USD": "$", "GBP": "£", "EUR": "€", "ILS": "₪"}
        currency_dictionary = {"₪": "ILS", "€": "EUR", "£": "GBP", "$": "USD"}

        numbers_dictionary = {"percent": "%", "Percent": "%", "percentage": "%", "Percentage": "%", "%": "%",
                              "Thousand": "K", "thousand": "K", "thousands": "K", "Thousands": "K", 'K': 'K',
                              "million": "M", "Million": "M", "Millions": "M", "millions": "M", "m": "M", 'M': 'M',
                              "Billion": "B", "billion": "B", "Billions": "B", "billions": "B", "b": "B", 'B': "B"}

        covid_list= ['COVID-19', 'COVID', 'COVID19', 'covid', 'Covid19', 'Covid', 'covid-19', 'Covid-19',
                            'COVID 19', 'covid 19', 'Corona', 'corona', 'coronavirus', 'Coronavirus', 'Corona virus',
                            'CoronaVirus', 'covid19', '#COVID19']

        USA_list = ['u.s.a', 'USA', 'usa', 'U.S', 'U.S.A', 'US', 'U.S.', 'u.s.', 'u.s', 'United States', '#USA']

        new_text = text.split()

        for word in new_text:
            # -- USA --
            if word in USA_list:
                if word == '#USA':
                    text += ' ' + 'USA'
                else:
                    text = text.replace(word, 'USA')

            if any(char.isdigit() for char in word) and not word.startswith('http'):
                add_char = ''
                if word.replace('.', '', 1).replace(',', '').replace('$', '').replace('£', '').replace('₪', '').replace('€', '').replace('+', '').isdigit():
                    if word.find('+') != -1:
                        add_char = '+'
                    if word.find('$') != -1:
                        add_char = '$'
                    if word.find('£') != -1:
                        add_char = '£'
                    if word.find('₪') != -1:
                        add_char = '₪'
                    if word.find('€') != -1:
                        add_char = '€'

                    old_word = word
                    word = word.replace(',', '').replace('$', '').replace('+', '').replace('₪', '').replace('£', '').replace('€', '')
                    # -- Find the length of the word --
                    number = word
                    lengthOfWord = number.find('.')
                    if lengthOfWord == -1:
                        lengthOfWord = len(number)

                    # -- Thousands --
                    if lengthOfWord > 3 and lengthOfWord < 7:
                        number = number[0:lengthOfWord]
                        number = number[0:lengthOfWord - 3] + '.' + number[lengthOfWord - 3:lengthOfWord]
                        if number[len(number) - 1] == "0": number = number[0:len(number) - 1]
                        word = number + "K"

                    # -- Millions --
                    if lengthOfWord > 6 and lengthOfWord < 10:
                        number = number[0:lengthOfWord]
                        number = number[0:lengthOfWord - 6] + '.' + number[lengthOfWord - 6:lengthOfWord - 3]
                        if number[len(number) - 1] == "0": number = number[0:len(number) - 1]
                        word = number + "M"

                    # -- Billions --
                    if lengthOfWord > 10:
                        number = number[0:lengthOfWord]
                        number = number[0:lengthOfWord - 9] + '.' + number[lengthOfWord - 9:lengthOfWord - 6]
                        if number[len(number) - 1] == "0": number = number[0:len(number) - 1]
                        word = number + "B"

                    pointIndex = number.find('.')
                    index = len(word) - 2
                    while pointIndex != -1 and index > pointIndex:
                        if word[index] == "0":
                            word = word[0:index - 1] + word[len(word) - 1]
                            index = index - 1
                        else:
                            break
                    if word[len(word) - 2] == '.':  word = word[0:len(word) - 2] + word[len(word) - 1]

                    text = text.replace(old_word, add_char + word)

        text_tokens = TweetTokenizer().tokenize(text)

        i = 0
        while i < len(text_tokens):
            if text_tokens[i] == "":
                i = i + 1
                continue

            if len(text_tokens[i]) > 1:
                while text_tokens[i].startswith(
                        ('/', '=', '-', '>', '<', '?', '_', '+', '-', '<', ')', ':', '(', '.', '~', '[', ']', ';')):
                    text_tokens[i] = text_tokens[i][1:]

            # --- numbers and percentage Parsing ---
            word = text_tokens[i]
            if i - 1 > -1 and word in numbers_dictionary:
                number = text_tokens[i - 1].replace(".", "")
                number = number.replace(',', "")
                if re.fullmatch(r'\d+', number) or re.fullmatch(r'\d+', number.replace('$', '').replace('₪', '').replace('£', '')):
                    text_tokens[i - 1] = text_tokens[i - 1] + numbers_dictionary.get(word)
                    del text_tokens[i]
                    continue


            if i + 1 < len(text_tokens) and text_tokens[i + 1].replace('/', '', 1).isdigit() and re.fullmatch(r'\d+',
                                                                                                              word):
                text_tokens[i] = text_tokens[i] + " " + text_tokens[i + 1]
                del text_tokens[i + 1]
                continue

            # -- Covid19 --
            word = text_tokens[i]
            if word in covid_list:
                if word == '#COVID19':
                    text_tokens.append('COVID19')
                else:
                    text_tokens[i] = 'COVID19'

                # -- "covid-19" or "covid 19"
                if i + 1 < len(text_tokens):
                    if text_tokens[i + 1] == '-':

                        if i + 2 < len(text_tokens) and text_tokens[i + 2] == '19':
                            del text_tokens[i + 1]
                            del text_tokens[i + 1]

                    # covid-19
                    elif text_tokens[i + 1] == '19':
                        del text_tokens[i + 1]

            # --- hashTag_Parsing ---
            word = text_tokens[i]

            if word.startswith("#"):
                token = re.compile("[A-Z][a-z]+|\d+|[A-Z][1-9]* +(?![a-z])").findall(word)

                capital_letters = [i for i, word in enumerate(word) if word.isupper()]
                if len(capital_letters) != 0:
                    token = [word[1: capital_letters[0]]] + token

                if len(capital_letters) == 0:
                    token = word[1:].split('_')

                token = [w.lower() for w in token]
                text_tokens[i] = text_tokens[i].lower()
                text_tokens.extend(token)

                i = i + 1
                continue

            # --- currency_Parsing ---
            word = text_tokens[i]
            if word in currency_dictionary:
                if i + 1 < len(text_tokens):
                    text_tokens[i + 1] = text_tokens[i + 1] + text_tokens[i]
                    del text_tokens[i]
                    continue

            if word in currency_sign_dictionary:
                if i - 1 > -1:
                    text_tokens[i - 1] = text_tokens[i - 1] + "" + currency_sign_dictionary[word]
                    del text_tokens[i]
                    continue

            # --- URL Parsing ---
            word = text_tokens[i]
            if word.startswith("http") or word.startswith("https"):
                if url != {}:
                    if word in url:
                        text_tokens[i] = url[word]

                        wordsList = re.split('\W+|\s', text_tokens[i])
                        text_tokens = text_tokens + wordsList[1:]

                del text_tokens[i]
                continue

            # --- Names and Entity ---
            word = text_tokens[i]
            if len(word) > 1 and word[0].isupper():
                j = i + 1
                name_or_entity = word
                flag = False
                while j < len(text_tokens):
                    if len(text_tokens[j]) > 1 and text_tokens[j][0].isupper():
                        name_or_entity += " " + text_tokens[j]
                        flag = True
                    else:
                        break
                    j += 1

                if flag:
                    if name_or_entity in self.names_and_entity_dictionary:
                        if tweet_id in self.names_and_entity_dictionary[name_or_entity]:
                            self.names_and_entity_dictionary[name_or_entity][tweet_id][0] += 1
                        else:
                            self.doc_name_and_entity.append(name_or_entity)
                            self.names_and_entity_dictionary[name_or_entity][tweet_id] = [1]
                    else:
                        self.doc_name_and_entity.append(name_or_entity)
                        self.names_and_entity_dictionary[name_or_entity] = {tweet_id: [1]}
                    i = j
                    continue

            i = i + 1
        words_to_remove = ["http", "https", "RT", "www", ":D", ":d", ";d", ":P", ":8", "__twitter_impression"]
        text_tokens = [w for w in text_tokens if w.lower() not in self.stop_words and w not in words_to_remove and (len(w) > 1 or (len(w) < 2 and w.isdigit()))]

        return text_tokens

    def remove_emoji(self, text):
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U00002500-\U00002BEF"  # chinese char
                                   u"\U00002702-\U000027B0"
                                   u"\U00002702-\U000027B0"
                                   u"\U000024C2-\U0001F251"
                                   u"\U0001f926-\U0001f937"
                                   u"\U00010000-\U0010ffff"
                                   u"\u2640-\u2642"
                                   u"\u2600-\u2B55"
                                   u"\u200d"
                                   u"\u23cf"
                                   u"\u23e9"
                                   u"\u231a"
                                   u"\ufe0f"  # dingbats
                                   u"\u3030"
                                   "\u0080-\U0010FFFF"
                                   "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)

    def parse_doc(self, doc_as_list):
        """
        This function takes a tweet document as list and break it into different fields
        :param doc_as_list: list re-preseting the tweet.
        :return: Document object with corresponding fields.
        """
        tweet_id = doc_as_list[0]
        full_text = doc_as_list[2]
        url = doc_as_list[3]

        # tweet_date = doc_as_list[1]
        # indices = doc_as_list[4]
        # retweet_text = doc_as_list[5]
        # retweet_url = doc_as_list[6]
        # retweet_indices = doc_as_list[7]
        # quote_text = doc_as_list[8]
        # quote_urls = doc_as_list[9]
        # quote_indices = doc_as_list[10]
        # retweet_quoted_text = doc_as_list[11]
        # retweet_quoted_urls = doc_as_list[12]
        # retweet_quoted_indices = doc_as_list[13]

        term_dict = {}

        try:
            if type(url) is str:
                url = url.replace("null", "\"null\"")
                url = eval(url)
        except:
            url = {}

        if full_text is not None:
            tokenized_text = self.parse_sentence(full_text, tweet_id, url)
        else:
            tokenized_text = None

        if self.to_stem:
            s = Stemmer()
            tokenized_text = [s.stem_term(token) for token in tokenized_text]

        max_tf = -1
        index = 0

        while index < len(tokenized_text):
            term = tokenized_text[index]
            letter = term[0].lower()
            if letter == '#' or letter == '@' or letter.isalnum():
                if term.lower() in tokenized_text:
                    term = term.lower()

                if term not in term_dict:
                    term_dict[term] = 1
                else:
                    term_dict[term] += 1

                if term_dict[term] > max_tf:
                    max_tf = term_dict[term]

            else:
                del tokenized_text[index]
                continue

            index += 1

        doc_length = len(tokenized_text)

        for key in self.doc_name_and_entity:
            if tweet_id in self.names_and_entity_dictionary[key]:
                self.names_and_entity_dictionary[key][tweet_id].append(max_tf)

        document = Document(tweet_id, tokenized_text, term_dict, doc_length, max_tf)
        return document
