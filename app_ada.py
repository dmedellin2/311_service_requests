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
    'open_data_channel_type' : user_input['open_data_channel_type']
    }

    descriptors = [
    "nr_descriptor",
    "ip_descriptor",
    "bd_descriptor",
    "nv_descriptor",
    "nss_descriptor",
    "av_descriptor",
    "nc_descriptor",
    "nepm_descriptor",
    "aa1_descriptor",
    "v_descriptor",
    "t_descriptor",
    "np_descriptor",
    "da_descriptor",
    "dv_descriptor",
    "d_descriptor",
    "g_descriptor",
    "nhow_descriptor",
    "aa2_descriptor",
    "dy_descriptor",
    "pa_descriptor"
    ]

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

    # fix complaint type values
    for key in renamed.keys():
        if key == data['complaint_type']:
            data['complaint_type'] = renamed[key]

    # find descriptor
    for descriptor in descriptors:
        if user_input[descriptor] != '':
            data['descriptor'] = user_input[descriptor]

    # make descriptors 'none' when necessary
    comp_no_descriptors = ['Homeless Encampment','Panhandling','Illegal Fireworks','Bike/Roller/Skate Chronic','Homeless Street Condition','Urinating in Public','Squeegee']

    if data['complaint_type'] in comp_no_descriptors:
        data['descriptor'] = 'none'

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

    # make dummy columns
    new_df_dummy = pd.get_dummies(new_df, columns=new_df.columns, drop_first=False)

    dummy_columns = [
    'complaint_type_Animal Abuse',
    'complaint_type_Animal-Abuse',
    'complaint_type_Bike/Roller/Skate Chronic',
    'complaint_type_Blocked Driveway',
    'complaint_type_Derelict Vehicle',
    'complaint_type_Disorderly Youth',
    'complaint_type_Drinking',
    'complaint_type_Drug Activity',
    'complaint_type_Graffiti',
    'complaint_type_Homeless Encampment',
    'complaint_type_Homeless Street Condition',
    'complaint_type_Illegal Fireworks',
    'complaint_type_Illegal Parking',
    'complaint_type_Noise - Commercial',
    'complaint_type_Noise - House of Worship',
    'complaint_type_Noise - Park',
    'complaint_type_Noise - Residential',
    'complaint_type_Noise - Street/Sidewalk',
    'complaint_type_Noise - Vehicle',
    'complaint_type_Non-Emergency Police Matter',
    'complaint_type_Panhandling',
    'complaint_type_Posting Advertisement',
    'complaint_type_Squeegee',
    'complaint_type_Traffic',
    'complaint_type_Urinating in Public',
    'complaint_type_Vending',
    'descriptor_Banging/Pounding',
    'descriptor_Blocked Bike Lane',
    'descriptor_Blocked Hydrant',
    'descriptor_Blocked Sidewalk',
    'descriptor_Building',
    'descriptor_Car/Truck Horn',
    'descriptor_Car/Truck Music',
    'descriptor_Chained',
    'descriptor_Chronic Speeding',
    'descriptor_Chronic Stoplight Violation',
    'descriptor_Commercial Overnight Parking',
    'descriptor_Congestion/Gridlock',
    'descriptor_Detached Trailer',
    'descriptor_Double Parked Blocking Traffic',
    'descriptor_Double Parked Blocking Vehicle',
    'descriptor_Drag Racing',
    'descriptor_Engine Idling',
    'descriptor_In Car',
    'descriptor_In Prohibited Area',
    'descriptor_In Public',
    'descriptor_Loud Music/Party',
    'descriptor_Loud Talking',
    'descriptor_Loud Television',
    'descriptor_Neglected',
    'descriptor_No Access',
    'descriptor_No Shelter',
    'descriptor_Nuisance/Truant',
    'descriptor_Other (complaint details)',
    'descriptor_Overnight Commercial Storage',
    'descriptor_Parking Permit Improper Use',
    'descriptor_Partial Access',
    'descriptor_Playing in Unsuitable Place',
    'descriptor_Police Report Not Requested',
    'descriptor_Police Report Requested',
    'descriptor_Posted Parking Sign Violation',
    'descriptor_Street Con Game',
    'descriptor_Ticket Scalping',
    'descriptor_Tortured',
    'descriptor_Trespassing',
    'descriptor_Truck Route Violation',
    'descriptor_Unauthorized Bus Layover',
    'descriptor_Underage - Licensed Est',
    'descriptor_Unlicensed',
    'descriptor_Use Indoor',
    'descriptor_Use Outside',
    'descriptor_Vehicle',
    'descriptor_With License Plate',
    'descriptor_none',
    'location_type_Club/Bar/Restaurant',
    'location_type_Commercial',
    'location_type_Common Area',
    'location_type_Hallway',
    'location_type_Highway',
    'location_type_House and Store',
    'location_type_House of Worship',
    'location_type_Lobby',
    'location_type_Other',
    'location_type_Park/Playground',
    'location_type_Parking Lot',
    'location_type_Residential Building',
    'location_type_Residential Building/House',
    'location_type_Roadway Tunnel',
    'location_type_Stairwell',
    'location_type_Store/Commercial',
    'location_type_Street/Sidewalk',
    'location_type_Subway',
    'location_type_Subway Station',
    'location_type_Vacant Lot',
    'location_type_unknown',
    'borough_BROOKLYN',
    'borough_MANHATTAN',
    'borough_QUEENS',
    'borough_STATEN ISLAND',
    'borough_Unspecified',
    'open_data_channel_type_ONLINE',
    'open_data_channel_type_OTHER',
    'open_data_channel_type_PHONE',
    'open_data_channel_type_UNKNOWN',
    'cluster_predicted_Noise Brooklyn',
    'cluster_predicted_Noise Queens',
    'cluster_predicted_Parking Manhattan',
    'cluster_predicted_Parking Queens'
    ]
    # drop "first" columns
    drop_columns = []

    for column in new_df_dummy.columns:
        if column not in dummy_columns:
            drop_columns.append(column)

    new_df_dummy.drop(columns=drop_columns, inplace=True)

    # add in missing columns
    for column in dummy_columns:
        if column not in new_df_dummy.columns:
            new_df_dummy.insert(loc=len(new_df_dummy.columns), column=column, value=0, allow_duplicates=False)

    # load model
    model = pickle.load(open("./models/ada.p", 'rb'))

    # make prediction
    pred = model.predict(new_df_dummy)[0]


    return render_template("confirmation.html", borough = data['borough'], location_type = data['location_type'], complaint_type = data['complaint_type'], descriptor = data['descriptor'], cluster = data['cluster_predicted'], open_data_channel_type = data['open_data_channel_type'], pred = round(pred,2))

# run the app
if __name__ == '__main__':
    app.run(debug = True)
