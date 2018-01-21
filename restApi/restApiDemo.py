from flask import Flask, jsonify, request
import pandas as pd
import nltk as nk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV

app = Flask(__name__)
global dataFrame
global grid_test
global y_test
global grid_predictions

def removePunctuation(sentence):
    r = [char for char in sentence if char not in string.punctuation]
    return ''.join(r)

def removeStopWords(sentence):
    return [word for word in sentence.split() if word.lower() not in stopwords.words('spanish')]

def cleanText(s):
    return removeStopWords(removePunctuation(s))

@app.route('/dataframe/all', methods=['GET'])
def get_dataframe():
    return dataframe.to_html()

@app.route('/model/params', methods=['GET'])
def get_bestparams():
    return jsonify(grid_test.best_params_)

@app.route('/model/report', methods=['GET'])
def get_report():
    print(classification_report(grid_predictions, y_test))

    return classification_report(grid_predictions, y_test)

@app.route('/model/predict', methods=['POST'])
def predict():
    if not request.json:
        abort(400)

    print(request.get_json()['message'])

    return 'CLASS: '+grid_test.predict([request.get_json()['message']]).tolist()[0]+'\r\n'

if __name__ == '__main__':
    dataframe = pd.read_excel('../tickets.xlsx')
    X_train, X_test, y_train, y_test = train_test_split(dataframe['descripcion'], dataframe['label'], test_size=0.35, random_state=42)

    predict_pipeline = Pipeline([
    ('BOW', CountVectorizer(analyzer=cleanText)),
    ('Tifid', TfidfTransformer()),
    ('Clasifier', MultinomialNB())
                            ])

    params = {'Tifid__use_idf':(True, False),
         'Clasifier__alpha': (1e-2, 1e-10),
         'BOW__ngram_range': [(1, 1), (1, 2)]}

    grid_test = GridSearchCV(predict_pipeline, params, verbose=3, n_jobs=-1)

    grid_test.fit(X_train, y_train)

    grid_predictions = grid_test.predict(X_test)

    app.run(debug=True)
