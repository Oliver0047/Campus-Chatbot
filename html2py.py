# -*- coding: UTF-8 -*-
#autor:Oliver
from flask import jsonify 
import json
from seq2seq import seq2seq
from flask import Flask, render_template, request
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('.')
chat=seq2seq()
app = Flask('chatbot')
@app.route('/')
def gethtml():   
    chat.prepare()
    return render_template('index.html')
@app.route('/predict',methods=['POST','GET'])
def predict():
    mydata = json.loads(request.get_data())
    data=mydata['question']
    try:
    	pred=chat.predict_one(data)
    except:
        chat.prepare()
        pred=chat.predict_one(data)
    return jsonify(result=pred)

if __name__=='__main__':
    app.run(host="0.0.0.0",port=5010)
