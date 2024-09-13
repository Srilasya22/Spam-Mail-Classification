from src.data_ingest import DataIngestion
from src.data_analysis import DataAnalysis
from src.text_preprocessing import TextPreprocessor
from src.feature_engineering import FeatureEngineering
from src.prepare_data import DatasetDevelopment
from src.model_training import ModelTraining
from src.evaluate import EvaluateModel
import joblib
import pandas as pd

def main():
    # Load the data
    data_ingest = DataIngestion("data/spam.csv")
    data = data_ingest.load_data()
    
    data_analysis = DataAnalysis(data)
    data_analysis.explore_data()
    data_ingest1 = DataIngestion("data/updated_data.csv")
    df= data_ingest1.load_data()
  
    text_preprocessor = TextPreprocessor()

    df['data'] = df['data'].apply(text_preprocessor._preprocess_text)
    
    feature_engineering = FeatureEngineering(df)
    feature_engineering.map_labels()
    print(df.head())

    data_dev = DatasetDevelopment(df)
    x_train, x_test, y_train, y_test = data_dev.divide_your_data()

    xtrain, xtest = feature_engineering.extract_features(x_train, x_test)
    print('Extracted features')
    model_train = ModelTraining(xtrain, y_train)
    model= model_train.Naive_Bayes()
    joblib.dump(model, "model.pkl")
    evaluate = EvaluateModel(xtest, y_test, model)
    evaluate.evaluate_model()


def predict(msg):
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    
    text_preprocessor = TextPreprocessor()
    preprocessed_msg = text_preprocessor._preprocess_text(msg)
    
    test_vector = vectorizer.transform([preprocessed_msg])

    output = model.predict(test_vector)
    
    return output[0]

if __name__ == "__main__":
    prediction = predict(
        "Hi, I am Lasya. I am a student of B.Tech. in Computer Science and Engineering."
    )
    if prediction == 0:
        print("Not a spam mail")
    else:
        print("Spam mail")