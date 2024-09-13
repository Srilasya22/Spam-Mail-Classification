from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

class EvaluateModel:
    def __init__(self, x_test, y_test, model):
        self.x_test = x_test
        self.y_test = y_test
        self.model = model

    def evaluate_model(self):
        print("Evaluating the model:- ")
        y_pred = self.model.predict(self.x_test)
        print("Accuracy Score:- ", accuracy_score(self.y_test, y_pred))
        print("Precision Score:- ", precision_score(self.y_test, y_pred))
        print("Recall Score:- ", recall_score(self.y_test, y_pred))
        print("F1 Score:- ", f1_score(self.y_test, y_pred))
        print("Completed evaluating the model")
