from datetime import datetime
import pandas as pd
import pickle
from geopy.geocoders import Nominatim
from shapely.geometry import Point, Polygon
from geopy import distance
from random import randint
import requests
import numpy  as np
import json
from datetime import timedelta


with open("pickup_cluster_model", 'rb') as file:
    pickup_kmean_model = pickle.load(file)

with open("destination_cluster_model", 'rb') as file:
    destination_kmean_model = pickle.load(file)

with open("pca_model", 'rb') as file:
    pca = pickle.load(file)

polygons = pickle.load( open( "DataFrame_for_maps", "rb" ) )

speed = pickle.load( open( "uber_speeds_aggregate.pickle", "rb" ) )
df_train = pickle.load( open( "train_data_ready", "rb" ) )





def predict_time(pickup_location, destination_location,day, month, year,pick_up_hour, pick_up_min) :


    day = int(day)
    month = int(month)
    year = int(year)
    pick_up_hour = int(pick_up_hour)
    pick_up_min = int(pick_up_min)
    platform_type = randint(0,4)


    if len(str(month)) == 1 :
        month = f'0{month}'

    date = f"{day}/{month}/{year}"
    date_time_obj = datetime.strptime(date, '%d/%m/%Y')
    day_of_week = date_time_obj.weekday()
    Personnal_or_Business = "Personnal"
    if Personnal_or_Business == "Personnal" :
        Personnal_or_Business = 1
    else :
        Personnal_or_Business = 0

    def is_week_end(day_of_week):
        if day_of_week > 5 :
            week_end = 1
        else:
            week_end = 0

        return week_end

    week_end = is_week_end(day_of_week)

    if (pick_up_hour >6 and pick_up_hour <9) or (pick_up_hour >16 and pick_up_hour <19) :
        rush_hour = 1
    else :
        rush_hour = 0



    geolocator = Nominatim(user_agent="patrick_app")
    pikcup_coord = geolocator.geocode(str(pickup_location))
    destination_coord = geolocator.geocode(str(destination_location))

    pickup_lat = pikcup_coord.latitude
    pickup_long = pikcup_coord.longitude


    destination_lat = destination_coord.latitude
    destination_long = destination_coord.longitude

    start = (pickup_lat, pickup_long)
    end  = (destination_lat, destination_long)

    distance_in_km = round(distance.distance(start, end).km  ,)

    pickup_cluster = pickup_kmean_model.predict(np.array([pickup_lat,pickup_long]).reshape(1,-1)).astype(str)
    destination_cluster = destination_kmean_model.predict(np.array([destination_lat,destination_long]).reshape(1,-1)).astype(str)

    pickup_cluster = pickup_kmean_model.predict(np.array([pickup_lat,pickup_long]).reshape(1,-1)).astype(str)
    destination_cluster = destination_kmean_model.predict(np.array([destination_lat,destination_long]).reshape(1,-1)).astype(str)

    long_lat = np.array([pickup_lat,pickup_long]).reshape(1,-1)

    pickup_pca_0 = pca.transform(np.array([pickup_lat,pickup_long]).reshape(1,-1))[:, 0]
    pickup_pca_1 = pca.transform(np.array([pickup_lat,pickup_long]).reshape(1,-1))[:, 1]
    destination_pca_0 = pca.transform(np.array([destination_lat,destination_long]).reshape(1,-1))[:, 0]
    destination_pca_1 = pca.transform(np.array([destination_lat,destination_long]).reshape(1,-1))[:, 1]

    area_pickup = 1
    for  row in polygons["geometry"] :
        area_pickup +=1
        if Point(pickup_lat, pickup_long).within(row) :
            pickup_id  = area_pickup

    destination_id = 0
    area_destination = 1
    for  row in polygons["geometry"] :
        area_destination +=1
        if Point(destination_lat, destination_long).within(row) :
            destination_id = area_destination

    std_travel_time = speed[(speed["sourceid"] == pickup_id) & (speed["dstid"] == destination_id) & (speed["hod"] == pick_up_hour) & (speed["week_end_or_not"] == is_week_end(day_of_week))]["std_travel_time"].values
    mean_travel_time = speed[(speed["sourceid"] == pickup_id) & (speed["dstid"] == destination_id) & (speed["hod"] == pick_up_hour) & (speed["week_end_or_not"] == is_week_end(day_of_week))]["mean_travel_time"].values

    new_dist_id = destination_id
    while len(std_travel_time.tolist()) == 0   :
        new_dist_id += 1
        std_travel_time = speed[(speed["sourceid"] == pickup_id) & (speed["dstid"] == new_dist_id ) & (speed["hod"] == pick_up_hour) & (speed["week_end_or_not"] == is_week_end(day_of_week))]["std_travel_time"].values

    while len(mean_travel_time.tolist()) == 0   :
        new_dist_id += 1
        mean_travel_time = speed[(speed["sourceid"] == pickup_id) & (speed["dstid"] == new_dist_id) & (speed["hod"] == pick_up_hour) & (speed["week_end_or_not"] == is_week_end(day_of_week))]["mean_travel_time"].values


    df_riders = pickle.load( open( "/Users/patrickmacclenihan/Desktop/final_project/DATA/riders_info_for_predict", "rb" ) )

    rider = pd.DataFrame(df_riders.iloc[randint(0,df_riders.shape[0]),1::]).T

    rider = rider.astype(int)

    url = "https://weatherbit-v1-mashape.p.rapidapi.com/current"

    querystring = {"lang":"en","lon":"36.8219462","lat":"-1.2920659"}

    headers = {
        'x-rapidapi-host': "weatherbit-v1-mashape.p.rapidapi.com",
        'x-rapidapi-key': "870480e1aamsh96e6e492e02203fp11faacjsndce42aecd5d0"
        }

    response = requests.request("GET", url, headers=headers, params=querystring)
    request = json.loads(response.text)
    temperature = request["data"][0]['temp']

    df_predict = pd.DataFrame(columns = ['Order No', 'User Id', 'Vehicle Type', 'Platform Type',
        'Personal or Business', 'Placement - Day of Month',
        'Placement - Weekday (Mo = 1)', 'Placement - Time',
        'Confirmation - Day of Month', 'Confirmation - Weekday (Mo = 1)',
        'Confirmation - Time', 'Arrival at Pickup - Day of Month',
        'Arrival at Pickup - Weekday (Mo = 1)', 'Arrival at Pickup - Time',
        'Pickup - Day of Month', 'Pickup - Weekday (Mo = 1)', 'Pickup - Time',
        'Arrival at Destination - Day of Month',
        'Arrival at Destination - Weekday (Mo = 1)',
        'Arrival at Destination - Time', 'Distance (KM)', 'Temperature',
        'Precipitation in millimeters', 'Pickup Lat', 'Pickup Long',
        'Destination Lat', 'Destination Long', 'Rider Id',
        'Time from Pickup to Arrival', 'No_Of_Orders', 'Age', 'Average_Rating',
        'No_of_Ratings', 'Pickup_Hour', 'Speed_in_km/h', 'Week_end',
        'pickup_long_lat', 'destination_long_lat', 'Pickup_Id',
        'Destination_Id', 'sourceid_x', 'dstid', 'hod_x', 'week_end_or_not_x',
        'std_travel_time', 'mean_travel_time', 'sourceid_y', 'hod_y',
        'week_end_or_not_y', 'min_travel_time', 'sourceid', 'hod',
        'week_end_or_not', 'min_std_travel_time', 'destination_clusters',
        'pickup_clusters', 'pickup_pca0', 'pickup_pca1', 'dropoff_pca0',
        'dropoff_pca1', 'Rush_Hour', 'Average_Rider_Speed'])

    for i in range(10) :
        df_predict["Placement - Weekday (Mo = 1)"] = day_of_week
        df_predict["Placement - Day of Month"] = day
        df_predict['Distance (KM)'] = distance_in_km
        df_predict["Platform Type"] = platform_type
        df_predict["Personal or Business"] = Personnal_or_Business
        df_predict["Pickup Lat"] = pickup_lat
        df_predict["Pickup Long"] = pickup_long
        df_predict["Destination Lat"] = destination_lat
        df_predict["Destination Long"] = destination_long
        df_predict["sourceid"] = pickup_id
        df_predict["dstid"] = destination_id
        df_predict["Pickup_Hour"] = pick_up_hour
        df_predict["Week_end"] = week_end
        df_predict['std_travel_time'] = std_travel_time
        df_predict["mean_travel_time"] = mean_travel_time
        df_predict['pickup_pca0'] = pickup_pca_0
        df_predict['pickup_pca1'] = pickup_pca_1
        df_predict['dropoff_pca0'] = destination_pca_0
        df_predict['dropoff_pca1'] = destination_pca_1
        df_predict['pickup_clusters'] = pickup_cluster
        df_predict["destination_clusters"] = destination_cluster
        df_predict['Rush_Hour'] = rush_hour
        df_predict["Temperature"] = temperature
        df_predict["Age"] = rider["Age"].values
        df_predict["Average_Rider_Speed"] = rider["Average_Rider_Speed"].values
        df_predict["No_Of_Orders"] = rider["No_Of_Orders"].values
        df_predict["Average_Rating"] = rider["Average_Rating"].values
        df_predict["No_of_Ratings"] = rider["No_of_Ratings"].values

    to_dummies = ['Platform Type',
    'Personal or Business',
    'Placement - Day of Month',
    'Pickup_Hour',
    'Pickup_Id',
    'Destination_Id',
    'Week_end',
    'destination_clusters',
    'pickup_clusters',
    'Rush_Hour',
    'Placement - Weekday (Mo = 1)']



    for col in to_dummies :
        col_val = df_predict[col][0]
        if col_val == 1 :
            df_predict.rename(columns={col:f'{col}_2'}, inplace=True)
            df_predict[f'{col}_2'] = 0
        else :
            df_predict.rename(columns={ col : f"{col}_{col_val}"},  inplace =True )
            df_predict[f"{col}_{col_val}"] = 1

    columns = pickle.load( open( "columns", "rb" ) )
    columns_to_drop = pickle.load( open( "columns_to_drop", "rb" ) )

    for col in columns :
        if col not in df_predict.columns:
            df_predict[col] = 0

    for col_to_drop in columns_to_drop :
        if col_to_drop in df_predict.columns :
            df_predict.drop(col_to_drop ,axis = 1 ,  inplace = True )

    for col in df_predict :
        if col not in columns :
            df_predict.drop(col, axis = 1)

    last_col_remove = []
    for i in df_predict :
        if i not in df_train :
            last_col_remove.append(i)

    while len(df_predict.columns) >= 300 :
        for col in last_col_remove :
            df_predict.drop(col, axis = 1, inplace = True)

    for i in df_predict :
        if i not in df_train :
            print(i)

    df_predict.fillna(0, inplace = True )


    model = pickle.load( open("model_finalized", "rb" ) )

    prediction = round(model.predict(df_predict).tolist()[0])
    delivery_time_date = datetime(int(year),int(month),int(day), int(pick_up_hour), int(pick_up_min))  +  timedelta(seconds= prediction )
    delivery_date = delivery_time_date.strftime("%m/%d/%Y")
    delivery_time = delivery_time_date.strftime("%H:%M")
    prediction_in_min = round(prediction / 60 , )

    return f"Your parcel will be deliver on : {delivery_date} at {delivery_time} - Duration : {prediction_in_min} minutes "
