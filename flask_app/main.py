from flask import Flask
from flask import Flask, render_template
from flask import request
import pandas as pd
import numpy as np
import torch
from transformers import RobertaForSequenceClassification
from transformers import AutoTokenizer
from module import MODEL, pre_treat

### pre_load ###
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = MODEL("/Users/damon/Documents/pre_onboarding/assignment3/checkpoint","klue/roberta-base")

app = Flask(__name__)

@app.route('/',methods = ['POST', 'GET'])
def index():
    return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
        
def Train():
    if request.method == 'POST':
 
        sentence1 = pre_treat(request.form['Sentence1'])
        sentence2 = pre_treat(request.form['Sentence2'])

        output = model.forward(request.form['Sentence1'],request.form['Sentence2'])
        #predition

        pred = output[0].item()
    return render_template('result.html', sen1=sentence1, sen2=sentence2, data1=pred)

if __name__ == '__main__': 
    app.run(debug=True)
