from collections import Counter
from flask import Flask, jsonify, render_template, request
from plotly.graph_objs import Bar, Heatmap
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import json
import joblib
import pandas as pd
import plotly
import string


app = Flask(__name__)

def tokenize(text):
    """
    Tokenize and lemmatize input text.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # define stop words and punctuation marks to exclude
    stop_words = set(stopwords.words('english'))
    exclude = set(string.punctuation)

    clean_tokens = []
    for tok in tokens:
        # convert token to lowercase and remove leading/trailing white space
        clean_tok = tok.lower().strip()

        # lemmatize token
        clean_tok = lemmatizer.lemmatize(clean_tok)

        # exclude stop words and punctuation marks
        if clean_tok not in stop_words and clean_tok not in exclude:
            clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages_categories', engine)

# load model
model = joblib.load("../models/classifier.pkl")

@app.route('/')
@app.route('/index')
def index():
    """
    Renders index page with visualizations.
    """
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = df.iloc[:, 4:].sum().sort_values(ascending=False)
    category_names = [cat.replace('_', ' ') for cat in list(df.iloc[:, 4:].columns)]

    genre_category_counts = df.groupby(['genre']).sum().iloc[:, 4:]


      # Top 20 most common words
    word_counter = Counter()
    for message in df.message:
        tokens = tokenize(message)
        word_counter.update(tokens)
    top_words = word_counter.most_common(20)[::-1]
    top_words_names = [word[0] for word in top_words]
    top_words_counts = [word[1] for word in top_words]


    # create visuals
    graphs = [
    {
        'data': [
            Bar(
                x=genre_names,
                y=genre_counts
            )
        ],
        'layout': {
            'title': 'Distribution of Message Genres',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Genre"
            }
        }
    },
    {
        'data': [
            Bar(
                x=category_names,
                y=category_counts
            )
        ],
        'layout': {
            'title': 'Distribution of Message Categories',
            'yaxis': {
                'title': "Count"
            },
            'xaxis': {
                'title': "Category"
            },
            'margin': {
                'b': 200
            }
        }
    },
    {
        'data': [
            Heatmap(
                x=genre_category_counts.columns,
                y=genre_category_counts.index,
                z=genre_category_counts.values,
                colorscale='YlOrRd'
            )
        ],
        'layout': {
            'title': {
                'text': 'Heatmap of Message Categories by Genre',
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'pad': {'b': 20},
            },
            'xaxis': {
                'title': 'Category',
                'title_standoff': 10,
            },
            'yaxis': {
                'title': 'Genre',
            },
            'margin': {'b': 200},
        }
    },
    {
        'data': [
            Bar(
                x=top_words_counts,
                y=top_words_names,
                orientation='h'
            )
        ],
        'layout': {
            'title': 'Top 20 Most Common Words',
            'yaxis': {
                'title': "Word",
                'tickfont': {'size': 10}  
            },
            'xaxis': {
                'title': "Count"
            },
            'height': 600  
        }
    }
]


# encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
