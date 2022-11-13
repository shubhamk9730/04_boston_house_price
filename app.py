from flask import Flask, render_template, request
import numpy as np
import pandas as pd

import pickle as pkl

model = pkl.load(open('model_DT_with_hyp.pkl', 'rb'))
app= Flask (__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict_class():
    CRIM = request.form['crim']
    ZN= request.form['zn']
    INDUS= request.form['indus']
    CHAS= request.form['chas']
    NOX= request.form['nox']
    RM= request.form['rm']
    AGE= request.form['age']
    DIS= request.form['dis']
    RAD= request.form['rad']
    TAX= request.form['tax']
    PTRATIO= request.form['ptratio']
    B= request.form['b']
    LSTAT= request.form['lstat']

    arr = np.array([[CRIM,ZN,INDUS,CHAS,NOX,RM,AGE,DIS,RAD,TAX,PTRATIO,B,LSTAT]])
    pred = str(model.predict(arr)[0])
    return pred

if __name__ == '__main__':
    app.run(debug=True)