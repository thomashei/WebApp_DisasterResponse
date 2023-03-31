# WebApp_DisasterResponse
## Project Overview

In this project, we will analyze disaster data from Appen to build a model for an API that classifies disaster messages. The dataset contains real messages that were sent during disaster events, and our goal is to create a machine learning pipeline to categorize these events so that we can send the messages to an appropriate disaster relief agency.

Our project includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. This project will show off our software skills, including our ability to create basic data pipelines and write clean, organized code!

The project has three main components:

ETL Pipeline
ML Pipeline
Flask Web App

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
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

### 3. Flask Web App
We provided much of the Flask web app, but we added extra features to show off our knowledge of Flask, HTML, CSS, and JavaScript. For this part, we did the following:

Modified file paths for database and model as needed
Added data visualizations using Plotly in the web app
Created two additional data visualizations based on data extracted from the SQLite database
