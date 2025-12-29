from flask import Flask,request,jsonify,render_template
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__);
app = application;

## Import redge regressor and standard scaler pickle

ridge_model = pickle.load(open('model/cross_validation_ridge.pkl','rb'));
scaler = pickle.load(open('model/scaler.pkl','rb'));

@app.route("/")
def index():
    return render_template('index.html');


@app.route("/predictdata", methods=['GET','POST'])
def predict_datapoint():
     if request.method == "POST":
        Temptature = float(request.form['Temperature']);
        RH = float(request.form['RH']);
        Ws = float(request.form['Ws']);
        Rain = float(request.form['Rain']);
        FFMC = float(request.form['FFMC']);
        DMC = float(request.form['DMC']);
        ISI = float(request.form['ISI']);
        Classes = request.form['Classes'];
        Region = request.form['Region'];
            
        new_scaled_data = scaler.transform([[Temptature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]]);
        result = ridge_model.predict(new_scaled_data);
     
        return render_template('home.html',result = result[0]);  
     
        

     else:
          return render_template('home.html');
     



if __name__ == "__main__":
     app.run(host="0.0.0.0");

    