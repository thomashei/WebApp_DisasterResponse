import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import pickle


def load_data(database_filepath):
    """
    Load data from database and split it into input X and output Y variables
    
    Args:
    database_filepath: str. File path of the database containing cleaned data
    
    Returns:
    X: dataframe. Feature dataframe containing messages
    Y: dataframe. Label dataframe containing category columns
    category_names: list. List of category names
    """
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('messages_categories', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    """
    Tokenize and clean the text data
    
    Args:
    text: str. Text data to be tokenized and cleaned
    
    Returns:
    tokens: list. List of cleaned and tokenized words
    """
    # normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # remove stopwords
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return tokens



def build_model():
    """
    Build a machine learning pipeline to classify messages into categories.
    Returns:
    pipeline (sklearn.pipeline.Pipeline): A pipeline object that preprocesses and classifies messages
    """


    pipeline = Pipeline([
    	('vect', CountVectorizer(tokenizer=tokenize)),
    	('tfidf', TfidfTransformer(use_idf=False)),
    	('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42, n_estimators=100, min_samples_split=3, class_weight='balanced')))
	])


    return pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the performance of a machine learning pipeline on test data.
    Args:
    model (sklearn.pipeline.Pipeline): A pipeline object that preprocesses and classifies messages
    X_test (pandas.DataFrame): A dataframe of test message data
    Y_test (pandas.DataFrame): A dataframe of test labels for each category
    category_names (list): A list of category names
    """
    # test the model
    Y_pred = model.predict(X_test)

    # print classification report
    for i, col in enumerate(category_names):
        print(col)
        try:
            print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
        except IndexError:
            print("No samples found for label:", col)



def save_model(model, model_filepath):
    """
    Save the trained model as a pickle file at the specified filepath.

    Args:
    model: Trained model object
    model_filepath (str): Filepath to save the trained model

    Returns:
    None
    """

    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
