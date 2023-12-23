from flask import Flask, render_template, jsonify, request
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot
import json

app = Flask(__name__)

# Load the model
regmodel = pickle.load(open("regmodel.pkl", "rb"))
scaler = pickle.load(open("scaling.pkl","rb"))

@app.route("/")
def home():
    return render_template("Home.html")


@app.route("/predict_api", methods=['POST'])
def predict_api():
    data = request.json["data"]
    print(data) 
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])
    

if __name__ == "__main__":
    app.run(debug=True)