import re
import csv
import sqlite3
import pandas as pd
import numpy as np
import string as str
import sklearn
import pickle
import tokenize


from flask import Flask, jsonify
from flask import request
from flasgger import Swagger, LazyString, LazyJSONEncoder
from flasgger import swag_from

# Import library for SKLearn Model Sentiment Analysis
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import train_test_split


from keras.preprocessing.text import Tokenizer
from keras.models import load_model
from keras.utils import pad_sequences


app = Flask(__name__)

app.json_encoder = LazyJSONEncoder
# app.json = LazyJSONEncoder
swagger_template = dict(
info = {
    'title': LazyString(lambda:'API Documentation for Data Processing and Modeling'),
    'version' : LazyString(lambda: '1.0.0'),
    'description' : LazyString(lambda: 'Dokumentasi API untuk Data Processing dan Modeling'),
    },
    host = LazyString(lambda: request.host)
)

swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'docs',
            "route": '/docs.json'
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/"
}

swagger = Swagger(app,template = swagger_template, config = swagger_config)

#Tools Function
max_features = 100000
sentiment = ['negative', 'neutral', 'positive']
tokenizer = Tokenizer(num_words=max_features, split=' ', lower=True)

# cnn
# file = open("C:/Users/62812/AppData/Local/Programs/Python/Python311/challange platinum/resources_of_cnn/x_pad_sequences.pickle", 'rb')
# feature_file_from_cnn = pickle.load(file)
# file.close()
# model_file_from_cnn = load_model("C:/Users/62812/AppData/Local/Programs/Python/Python311/challange platinum/model_of_cnn/model.h5")

# lstm
file = open("Tugas Platinum Binar Academy/Model LSTM/x_pad_sequences.pickle","rb")
feature_file_from_lstm = pickle.load(file)
file.close()
model_file_from_lstm = load_model("Tugas Platinum Binar Academy/Model LSTM/model.h5")


#Vectorizer For Neural Network
count_vect = pickle.load(open("Tugas Platinum Binar Academy/Model NN/feature.p","rb"))

#Load Model for Neural Network
model_NN = pickle.load(open("Tugas Platinum Binar Academy/Model NN/model.p","rb"))

# Homepage
@app.route('/', methods=['GET'])
def get():
    return "WELCOME TO MYWORKSSS" 

@swag_from("docs/nnt.yml", methods=['POST'])
@app.route('/NN_Text', methods=['POST'])
def text_processing():
    teks_input = request.form.get('text')
    teks_output = cleansing(teks_input)
    #Vectorizing 
    text = count_vect.transform([teks_output])
     #Predict sentiment
    result = model_NN.predict(text)[0]

    json_respon = {
        'input' : teks_input,
        'sentiment' : result,
        'output' : teks_output
    }
    response_data = jsonify(json_respon)
   
    return response_data

# @swag_from("docs/cnn.yml", methods=['POST'])
# @app.route('/cnn', methods=['POST'])
# def cnn():
#     original_text = request.form.get('text')
#     text = [cleansing(original_text)]
#     feature = tokenizer.texts_to_sequences(text)
#     feature = pad_sequences(feature, maxlen=feature_file_from_cnn.shape[1])
#     prediction = model_file_from_cnn.predict(feature)
#     get_sentiment = sentiment[np.argmax(prediction[0])]

#     json_response = {
#         'status_code': 200,
#         'description': "Result of Sentiment Analysis using CNN",
#         'data': {
#             'text' : original_text,
#             # 'sentiment' : prediction
#             'sentiment' : get_sentiment
#         },
#     }

#     response_data = jsonify(json_response)
#     return response_data

@swag_from("docs/lstmt.yml", methods=['POST'])
@app.route('/lstm', methods=['POST'])
def lstm():
    original_text = request.form.get('text')
    text = [cleansing(original_text)]
    feature = tokenizer.texts_to_sequences(text)
    feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
    
    prediction = model_file_from_lstm.predict(feature)
    get_sentiment = sentiment[np.argmax(prediction[0])]

    json_response = {
        'status_code': 200,
        'description': "Result of Sentiment Analysis using LSTM",
        'data': {
            'text' : original_text,
            # 'sentiment' : prediction
            'sentiment' : get_sentiment
        },
    }

    response_data = jsonify(json_response)
    return response_data


@swag_from("docs/nnf.yml", methods=['POST'])
@app.route('/NN_Upload_File', methods=['POST'])
def NN_Upload_File():
    file = request.files["file"]
    df_csv = pd.read_csv(file, encoding="latin-1")
    df_csv = df_csv['data']
    df_csv = df_csv.tail()
    df_csv = df_csv.drop_duplicates()
    df_csv = df_csv.values.tolist()

    ix = 0
    datanya = {}
    for str in df_csv:
        # text = ''
        # feature = ''
        # prediction = ''
        # get_sentiment = ''
        # # text = [cleansing(str)]
        # feature = tokenizer.texts_to_sequences(text)
        # feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
        # prediction = model_file_from_lstm.predict(feature)
        # get_sentiment = sentiment[np.argmax(prediction[0])]

        teks_output = cleansing(str)
        #Vectorizing 
        text = count_vect.transform([teks_output])
        #Predict sentiment
        result = model_NN.predict(text)[0]


        datanya[ix] = {}
        datanya[ix]['text'] = teks_output
        datanya[ix]['sentiment'] = result
        ix = ix + 1
    # return_file = {
    #     'output' : df_csv
    # }
    return datanya


@swag_from("docs/lstmf.yml", methods=['POST'])
@app.route('/Lstm_Upload_File', methods=['POST'])
def lstm_upload_file():
    file = request.files["file"]
    df_csv = pd.read_csv(file, encoding="latin-1")
    df_csv = df_csv['data']
    df_csv = df_csv.tail()
    df_csv = df_csv.drop_duplicates()
    df_csv = df_csv.values.tolist()
    ix = 0
    datanya = {}
    for str in df_csv:
        text = ''
        feature = ''
        prediction = ''
        get_sentiment = ''
        text = [cleansing(str)]
        feature = tokenizer.texts_to_sequences(text)
        feature = pad_sequences(feature, maxlen=feature_file_from_lstm.shape[1])
        prediction = model_file_from_lstm.predict(feature)
        get_sentiment = sentiment[np.argmax(prediction[0])]
        datanya[ix] = {}
        datanya[ix]['text'] = text
        datanya[ix]['sentiment'] = get_sentiment
        ix = ix + 1
    # return_file = {
    #     'output' : df_csv
    # }
    return datanya




# def file_processing():
#     #upload file
#     file = request.files['file']
#     #Import file to pandas DataFrame
#     df = pd.read_csv(file, encoding="latin-1")
    
#     # Define text preprocessing function
#     def cleansing(text):
#         return text
    
#     # Cleanse text
#     df['text_clean'] = df.apply(lambda row : cleansing(row['text']), axis = 1)
    
#     # Initialize vectorizer object
#     count_vect = CountVectorizer()

#     # Fit vectorizer object to text data
#     count_vect.fit(df['text_clean'])

#     result = []
#     #Vectorizing & Predict sentiment
#     for index, row in df.iterrows():
#         text = count_vect.transform([row['text_clean']])

#         #append predicted sentiment to result 
#         result.append(model_NN.predict(text)[0])
        
#     # Get result from file in "List" format
#     original_text = df['text_clean'].to_list()

#     json_respon = {
#         'status_code': 200,
#         'description': "Result of Sentiment Analysis using Neural Network",
#         'data': {
#         'text' : original_text,
#         'sentiment' : result
#         },
#     }

#     response_data = jsonify(json_respon)
#     return response_data


def removeVowels(str):
    vowels = 'aeiou'
    for ele in vowels:
        str = str.replace(ele, 'x')
    return str

#===============================================================================
#cleansing
def case_folding(teks):
   # proses cleansing
    # 1. jadikan teks agar lowercase
    teks = teks.lower()
    # 2. hanya menerima alfabet text
    teks = re.sub(r'[^a-zA-Z]',' ', teks)
    teks = re.sub(r':', '', teks)
    teks = re.sub('\n',' ',teks)
    teks = re.sub('rt',' ', teks)
    teks = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ', teks)
    teks = re.sub('  +', ' ', teks)
    teks = re.sub(r'pic.twitter.com.[\w]+', '', teks)
    teks = re.sub('user',' ', teks)
    teks = re.sub('gue','saya', teks)
    teks = re.sub(r'‚Ä¶', '', teks)
    # 3. hapus kata user
    teks = teks.replace("user", "")
    # 4. menghapus whitespace di awal dan di akhir kalimat
    teks = teks.strip()
    # 5.hapus kata yg hanya terdiri dari 2 huruf atau kurang
    teks = teks.split()
    teks_normal = ''
    for str in teks:
        if(len(str) > 2):
            if(teks_normal == ''):
                teks_normal = str
            else:
                teks_normal = teks_normal + ' ' + str
    return teks_normal

def cleansing(teks):
    teks = case_folding(teks)
    
    return teks

if __name__ == '__main__' :
    app.run(debug=True)
