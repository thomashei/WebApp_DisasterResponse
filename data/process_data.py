import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load and merge the messages and categories datasets
    
    Args:
    messages_filepath: str, filepath of the messages dataset
    categories_filepath: str, filepath of the categories dataset
    
    Returns:
    df: pandas DataFrame, merged dataset of messages and categories
    
    """
    # Load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    """
    Clean and preprocess the merged dataset
    
    Args:
    df: pandas DataFrame, merged dataset of messages and categories
    
    Returns:
    df: pandas DataFrame, cleaned and preprocessed dataset
    
    """
    # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Rename the columns of `categories`
    category_colnames = categories.iloc[0].apply(lambda x: x[:-2]).values.tolist()
    categories.columns = category_colnames
    
    # Convert category values to numeric
    for column in categories:
        categories[column] = categories[column].apply(lambda x: int(x[-1]))

    # Remove rows with category values other than 0 and 1
    categories = categories.loc[~(categories == 2).any(axis=1)]
    
    # Replace categories column in df with new category columns
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # Drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    Save the cleaned dataset to a sqlite database
    
    Args:
    df: pandas DataFrame, cleaned dataset
    database_filename: str, filename of the database to save to
    
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages_categories', engine, index=False, if_exists='replace')  


def main():
    """
    Executes the ETL pipeline to preprocess data and save to database
    
    """
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        print('Cleaning data...')
        df = clean_data(df)
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        print('Cleaned data saved to database!')
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
