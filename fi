import spacy
import nltk
import pandas as pd
import sklearn as sk
import numpy as np
#import wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
#import en_core_web_sm
import matplotlib.pyplot as plt
#import itertools
#nlp = spacy.load("en_core_web_sm")
#doc = nlp(text)
#tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
#nltk.download('stopwords')
#spacy_stopwords = spacy.en.stop_words.STOP_WORDS
nltk_stopwords = nltk.corpus.stopwords.words('english')
#train_df = pd.DataFrame(columns=['Id','Contents','Label'])
#test_df = pd.DataFrame( columns=['Id', 'Contents'])
#answer_df = pd.DataFrame( columns=['Id', 'Label'])
def Print_Article(article):
    pass
def NormalCountSentences(train_df):
    count = 0
    for index, row in train_df.iterrows():
        if row["Label"]==1: # the news is fake
            count+=1
    print("There are {} normal articles ".format(count))
def FakeCountSentences(train_df):
    count = 0
    for index, row in train_df.iterrows():
        if row["Label"]==0: # the news is fake
            count+=1
    print("There are {} fake articles ".format(count))
def process_Article(article):
    pass
def ListToString(lis):
    for x in lis:
        str = str + x + " "
    return str
 
def Read_Data():
    train_df = pd.read_csv("textract_train.csv")
    test_df = pd.read_csv("textract_test.csv")
    #answer_df
    train_size = len(train_df.index)
    #NormalCountSentences(train_df)
    #FakeCountSentences(train_df)
    """
    for index, row in train_df.iterrows():
        if row['Id']==50:
            break
        if row["Label"]==1: # the news is fake
            #black is fake
            article = row["Contents"]
            wordcloud = WordCloud(stopwords=nltk_stopwords,background_color="black").generate(article)
            #black for fake
            # Display the generated image:
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            strin = "black_review " + str(row['Id']) + ".png"
            wordcloud.to_file(strin)
            #break
            #print_Article(article)
            #process_Article(article)
            #Print_Article(df[i][1])
       
        else:#white is normal
            article = row["Contents"]
            wordcloud = WordCloud(stopwords=nltk_stopwords,background_color="white").generate(article)
            #white for real
            # Display the generated image:
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            strin = "white_review " + str(row['Id'])  + ".png"
            wordcloud.to_file(strin)
            #break
    """
    X_train = train_df['Contents']
    y_train = train_df['Label']
    #X_test = test_df['Contents']
 
    pipe = Pipeline([('cvec', CountVectorizer()),
                     ('nb', MultinomialNB())])
    # Tune GridSearchCV
    pipe_params = {'cvec__ngram_range': [(1, 1), (1, 3)],
                   'nb__alpha': [.36, .6]}
    gs = GridSearchCV(pipe, param_grid=pipe_params, cv=3)
    gs.fit(X_train, y_train);
 
    print("Best score:", gs.best_score_)
    print("Train score", gs.score(X_train, y_train))
    print("a")
    #print("Test score", gs.score(X_test, y_test))
 
    """
    pipe = Pipeline([('cvec', CountVectorizer()),
                     ('lr', LogisticRegression(solver='liblinear'))])
    # Tune GridSearchCV
    pipe_params = {'cvec__stop_words': [None, 'english', custom],
                   'cvec__ngram_range': [(1, 1), (2, 2), (1, 3)],
                   'lr__C': [0.01, 1]}
    gs = GridSearchCV(pipe, param_grid=pipe_params, cv=3)
    gs.fit(X_train, y_train);
    print("Best score:", gs.best_score_)
    print("Train score", gs.score(X_train, y_train))
    print("Test score", gs.score(X_test, y_test))
    """
    """
    answer_df = pd.DataFrame(data, columns=['Id', 'Label'])
    for index, row in answer_df.iterrows():
    """
 
def Main():
    Read_Data()
if __name__=="__main__":
    Main()
