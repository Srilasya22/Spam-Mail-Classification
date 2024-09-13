from sklearn.model_selection import train_test_split
import pandas as pd




class DatasetDevelopment:
    def __init__(self, df):
        self.df = df

    def divide_your_data(self):
        print("Dividing the data:- ")

        X = self.df['data']
        y = self.df["labels"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2,random_state=42
        )
        return X_train, X_test, y_train, y_test