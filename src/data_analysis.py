import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt 
from wordcloud import WordCloud, STOPWORDS
import nltk  

nltk.download("stopwords")
stopwords = set(STOPWORDS)

class DataAnalysis:
    def __init__(self, df) -> None:
        self.df = df
        self.clean_data()  # Automatically clean the data upon initialization

    def explore_data(self):
        print("Head of the data:\n", self.df.head())
        print("Shape of the data:", self.df.shape)

        # Visualize the distribution of labels
        self.visualize_label_distribution()

        # Create a word cloud of the 'data' column
        self.create_word_cloud()

    def clean_data(self):
        # Dropping unnecessary columns if they exist
        columns_to_drop = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']
        self.df = self.df.drop(columns=[col for col in columns_to_drop if col in self.df.columns], axis=1)
        # Renaming columns
        self.df.columns = ['labels', 'data'] 
        self.df.to_csv('data/updated_data.csv')

    def visualize_label_distribution(self):
        """Visualize the distribution of labels."""
        label_counts = self.df['labels'].value_counts()
        plt.figure(figsize=(6, 4))
        label_counts.plot(kind='bar', color=['skyblue', 'salmon'])
        plt.title('Distribution of Labels (Spam vs Not Spam)')
        plt.xlabel('Labels')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.show()

    def create_word_cloud(self):
        """Create and display a word cloud for the 'data' column."""
        all_text = ' '.join(self.df['data'].tolist()) 
        wordcloud = WordCloud(stopwords=stopwords, background_color='white', max_words=200, width=800, height=400).generate(all_text)

        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud of Spam Messages')
        plt.show()
