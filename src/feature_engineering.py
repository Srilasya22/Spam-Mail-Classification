import nltk
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib

class FeatureEngineering:

    """ 
    Feature Engineering
     Mapping Labels   
    """

    def __init__(self, df):
        self.df = df

    def map_labels(self):
        self.df["labels"] = self.df.labels.map({"ham": 0, "spam": 1})

        return self.df
    
    def extract_features(self,x_train,x_test):
        vectorizer = CountVectorizer()
        xtrain= vectorizer.fit_transform(x_train)
        xtest=vectorizer.transform(x_test)
        joblib.dump(vectorizer, "vectorizer.pkl")
        return xtrain,xtest