import pandas as pd  
import numpy as np 
import re  
import nltk 

from nltk.corpus import stopwords
import multiprocessing as mp
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    def __init__(self):
        """
        Text preprocessing transformer includes steps:
            1. Text normalization
            2. Punctuation removal
            3. Stop words removal
        """

    def _preprocess_text(self, text):
        text = text.lower()
        text = self._remove_punctuations(text)
        text = self._remove_stop_words(text)
        text = self._remove_urls(text)
        text = self._remove_numbers(text)
        text = self._stem_text(text)
        return text

    def _remove_punctuations(self, text):
        # function for removing punctuations
        import string

        text = text.translate(str.maketrans("", "", string.punctuation))
        return text

    def _remove_stop_words(self, text):
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words("english"))
        words = text.split()
        words = [w for w in words if not w in stop_words]
        text = " ".join(words)
        return text

    def _remove_urls(self, text):
        text = re.sub(r"http\S+", "", text)
        return text

    def remove_hashtags(self, text):
        text = re.sub(r"#\S+", "", text)
        return text

    def _remove_numbers(self, text):
        text = re.sub(r"\d+", "", text)
        return text

    def _stem_text(self, text):
        # function for stemming text

        stemmer = PorterStemmer()
        text = " ".join([stemmer.stem(word) for word in text.split()])
        return text

    def _lemmatize_text(self, text):
        # function for lemmatizing text
        lemmatizer = WordNetLemmatizer()
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        return text
