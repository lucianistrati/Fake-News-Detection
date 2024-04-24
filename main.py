import spacy
import nltk
import pandas as pd
import sklearn as sk
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

nltk.download('stopwords')
nltk_stopwords = nltk.corpus.stopwords.words('english')

def print_article(article):
    pass

def normal_count_sentences(train_df):
    count = 0
    for index, row in train_df.iterrows():
        if row["Label"] == 1: 
            count += 1
    print("There are {} normal articles ".format(count))

def fake_count_sentences(train_df):
    count = 0
    for index, row in train_df.iterrows():
        if row["Label"] == 0: 
            count += 1
    print("There are {} fake articles ".format(count))

def process_article(article):
    pass

def list_to_string(lis):
    string = ""
    for x in lis:
        string += x + " "
    return string

def read_data():
    train_df = pd.read_csv("textract_train.csv")
    test_df = pd.read_csv("textract_test.csv")
    X_train = train_df['Contents']
    y_train = train_df['Label']

    pipe = Pipeline([('cvec', CountVectorizer()),
                     ('nb', MultinomialNB())])
    
    pipe_params = {'cvec__ngram_range': [(1, 1), (1, 3)],
                   'nb__alpha': [.36, .6]}
    gs = GridSearchCV(pipe, param_grid=pipe_params, cv=3)
    gs.fit(X_train, y_train)

    print("Best score:", gs.best_score_)
    print("Train score", gs.score(X_train, y_train))

def main():
    read_data()

if __name__=="__main__":
    main()
