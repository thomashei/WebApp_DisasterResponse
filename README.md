# WebApp_DisasterResponse
## Project Overview

In this project, we will analyze disaster data from Appen to build a model for an API that classifies disaster messages. The dataset contains real messages that were sent during disaster events, and our goal is to create a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

Our project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off our software skills, including our ability to create basic data pipelines and write clean, organized code!

The project has three main components:

1. ETL Pipeline
2. ML Pipeline
3. Flask Web App


## Project Components
### 1. ETL Pipeline
In the Python script, process_data.py, we wrote a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database

### 2. ML Pipeline
In the Python script, train_classifier.py, we wrote a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Outputs results on the test set
Exports the final model as a pickle file

### 3. Flask Web App
In the Python script, run.py, once you pass the file paths of database and model as needed, the app allows you to input a new message and get classification results in several categories. 

The web app will also display visualizations of the data. On top of the visual which is already provided, we added extra visuals to show the relationship between message categories and genres, and also the top 20 word count of disaster messages.


## File structure and description

| Directory/File | Description |
| --- | --- |
| app/ | Directory containing Flask application |
| app/template/ | Directory containing HTML templates |
| app/template/master.html | Main page of web app |
| app/template/go.html | Classification result page of web app |
| app/run.py | Flask file that runs app |
| data/ | Directory containing data files |
| data/disaster_categories.csv | Data to process |
| data/disaster_messages.csv | Data to process |
| data/process_data.py | Python script for ETL pipeline |
| data/InsertDatabaseName.db | SQLite database to save clean data to |
| models/ | Directory containing machine learning models |
| models/train_classifier.py | Python script for machine learning pipeline |
| models/classifier.pkl | Saved model |


## How to run

Run the following commands in the project's root directory to set up your database and model.

### 1. To run ETL pipeline that cleans data and stores in database
    `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
       
### 2. To run ML pipeline that trains classifier and saves
    `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`


Then go to `app` directory: `cd app`

### 3. Run your web app: 
    `python run.py`
