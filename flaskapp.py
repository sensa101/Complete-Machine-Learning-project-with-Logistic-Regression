from flask import Flask,request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
app=Flask(__name__)
Swagger(app)
pickle_in=open('iris_clf_model.pkl','rb')
model=pickle.load(pickle_in)
@app.route('/predict',methods=['Get'])
def predict_class():
    sw=float(request.args.get("sw"))
    pw=float(request.args.get("pw"))
    encoded_predicted_label=model.predict([[sw,pw]])[0]
    original_label_name=LE.inverse_transform([encoded_predicted_label])[0]
    return "Model prediction is "+original_label_name)

if __name__=='__main__':
            app.run(debug=True,host='0.0.0.0')

