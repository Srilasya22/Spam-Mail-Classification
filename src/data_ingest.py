import pandas as pd

class DataIngestion:
    """
     Class for ingesting data
    """
    def __init__(self,file_path):
        """
        Initializes the class with file path
        :param file_path:str path to csv file
        """
        self.file_path=file_path

    def load_data(self):
        """
        Loads the data from csv file 
        :returns data:pandas dataframe,data loaded from csv file
        """
        with open(self.file_path, 'r', encoding='utf-8', errors='ignore') as file:
            data = pd.read_csv(file)
        print("Data loaded")
        return data
