# imports
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, Response, render_template, jsonify

# initialize the flask app
app = Flask("myApp")

### route 1: hello world
# define the route
@app.route("/") # home route
# create the controller
def home():
    return "This is the home page HTML?"

## route 2: show a form to user
@app.route("/form")
def form():
    # use flask's render_template
    return render_template("form.html")

## route 3: accept form submission & predict
@app.route("/submit")
def make_prediction():
    # load user data
    user_input = request.args
    data = np.array([
    int(user_input['OveralQual']),
    int(user_input['FullBath']),
    int(user_input['GarageArea']),
    int(user_input['LotArea'])
    ]).reshape(1, -1)

    # load model
    model = pickle.load(open("./model/model.p", 'rb'))

    # make prediction
    pred = model.predict(data)[0]


    return render_template("confirmation.html", prediction = round(pred, 2))

# run the app
if __name__ == '__main__':
    app.run(debug = True)
