!pip install kmodes

# imports
import pickle
import numpy as np
import pandas as pd
from kmodes.kmodes import KModes
from flask import Flask, request, Response, render_template, jsonify

# initialize the flask app
app = Flask("myApp")

### route 1: hello world
# define the route
@app.route("/") # home route
def form():
    # use flask's render_template
    return render_template("form.html")

## route 3: accept form submission & predict
@app.route("/submit")
def make_prediction():
    # load user data
    user_input = request.args
    data = {
    'borough':user_input['borough'],
    'location_type' : user_input['location_type'],
    'complaint_type': user_input['complaint_type'],
    'descriptor' : user_input['descriptor'],
    'open_data_channel_type' : 'ONLINE'
    }

    renamed = {
    'Noise-Residential':'Noise - Residential',
    'IllegalParking':'Illegal Parking',
    'BlockedDriveway' :'Blocked Driveway',
    'Noise-StreetSidewalk':'Noise - Street/Sidewalk',
    'Noise-Vehicle':'Noise - Vehicle',
    'AbandonedVehicle':'Abandoned Vehicle',
    'Noise-Commercial':'Noise - Commercial',
    'Non-EmergencyPoliceMatter': 'Non-Emergency Police Matter',
    'HomelessEncampment' : 'Homeless Encampment',
    'Animal-Abuse' :'Animal-Abuse',
    'Vending':'Vending',
    'Traffic' : 'Traffic',
    'Noise-Park':'Noise - Park',
    'Panhandling':'Panhandling',
    'DrugActivity' : 'Drug Activity',
    'DerelictVehicle':'Derelict Vehicle',
    'Drinking':'Drinking',
    'IllegalFireworks':'Illegal Fireworks',
    'Bike/Roller/Skate Chronic':'Bike/Roller/Skate Chronic',
    'Graffiti':'Graffiti',
    'HomelessStreetCondition':'Homeless Street Condition',
    'Noise-HouseofWorship':'Noise - House of Worship',
    'UrinatinginPublic':'Urinating in Public',
    'AnimalAbuse':'Animal Abuse',
    'DisorderlyYouth':'Disorderly Youth',
    'PostingAdvertisement':'Posting Advertisement',
    'Squeegee':'Squeegee'
    }

    for key in renamed.keys():
        if key == data['complaint_type']:
            data['complaint_type'] = renamed[key]

    df = pd.DataFrame(columns=data.keys())
    df = df.append(data, ignore_index=True)

    # load k-modes
    km = pickle.load(open("./models/km.p", 'rb'))
    # predict cluster
    cluster = km.predict(df)
    # add cluster to data
    data['cluster_predicted'] = cluster[0]
    # rename cluster
    cluster_names = {
    0: 'Noise Brooklyn',
    1: 'Noise Queens',
    2: 'Parking Queens',
    3: 'Driveway Queens',
    4: 'Parking Manhattan'
    }

    for key in cluster_names.keys():
        if key == data['cluster_predicted']:
            data['cluster_predicted'] = cluster_names[key]

    # new df with cluster
    new_df = pd.DataFrame(columns=data.keys())
    new_df = new_df.append(data, ignore_index=True)


    # make prediction
    #pred = model.predict(data)[0]


    return render_template("confirmation.html", borough = data['borough'], location_type = data['location_type'], complaint_type = data['complaint_type'], descriptor = data['descriptor'], cluster= data['cluster_predicted']) #, prediction = round(pred, 2))

# run the app
if __name__ == '__main__':
    app.run(debug = True)
