# prompt: generate code to Load model and vectorizer to predict the previous datapoint
from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import virtualenv
import os
app = Flask(__name__, template_folder='templates')

# Load the trained model and vectorizer
with open('model.pkl', 'rb') as model_file:
    clf = pickle.load(model_file)

# Load the vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)
    
# Use Stemming    
port_stem=PorterStemmer()
def stemming(content):
        stemmed_content=re.sub('[^a-zA-Z]',' ',content)
        stemmed_content=stemmed_content.lower()
        stemmed_content=stemmed_content.split()
        '''
        for word in (stemmed_content):
                if word not in stopwords.words('english'):
                        stemmed_content=port_stem.stem(word)
        '''            
        #stemmed_content=[port_stem.stem(word) for word in (stemmed_content) if word not in stopwords.words('english')]
        stemmed_content=' '.join(stemmed_content)
        #print(stemmed_content)
        return stemmed_content

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/FakeNews_test', methods=['POST'])
def predict():
    news = request.form['news']
    testing_news={"text":[news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test["text"]=new_def_test["text"].apply(stemming)
    new_x_test=new_def_test["text"]
    #print (new_x_test)
    new_xv_test=vectorizer.transform(new_x_test)
    test_res=clf.predict(new_xv_test)
    if test_res[0]==0:
            result="Fake News"
    else:
            result="Authentic News"
        
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    
    app.run(debug=True, host='0.0.0.0', port=5000)
