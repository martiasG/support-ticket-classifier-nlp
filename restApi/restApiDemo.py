from flask import Flask, jsonify
import pandas as pd
import nltk as nk
import seaborn as sns
import matplotlib.pyplot as plt
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from nltk.stem import SnowballStemmer
#Add this imports
from io import StringIO
import base64

app = Flask(__name__)

@app.route('/dataframe/all', methods=['GET'])
def get_dataframe():
    df = pd.read_excel('../ticketsSupportxls')
    img = StringIO()
    y = [1,2,3,4,5]
    x = [0,2,1,3,4]

    df.hist(column='label', by='Estado', figsize=(14, 8), bins=50)
    plt.plot(x,y)
    plt.savefig(img, format='png')
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue())

    return render_template('template_graph.html', plot_url=plot_url)

if __name__ == '__main__':
    app.run(debug=True)
