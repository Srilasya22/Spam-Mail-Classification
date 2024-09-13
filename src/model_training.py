from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

class ModelTraining:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def logistic_regression_model(self):
        print("Training the model logistic regression model:- ")
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(self.x_train, self.y_train)
        
        print("Completed training the model")
        return log_reg


    def Naive_Bayes(self):
        print("Training the model Naive Bayes:- ")
        nb_model = MultinomialNB()
        nb_model.fit(self.x_train, self.y_train)
        print("Completed training the model")
        return nb_model

   