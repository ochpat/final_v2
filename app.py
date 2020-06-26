from dash.dependencies import Input, Output,  State
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import pickle
import plotly.express as px
import dash_table
import seaborn as sns
import base64
from predict import predict_time
from twilio.rest import Client
import folium
from geopy.geocoders import Nominatim
from datetime import datetime



path_train = "Train.csv"
path_riders = "Riders.csv"
df_train = pd.read_csv(path_train)
df_riders = pd.read_csv(path_riders)
df_speed = pickle.load( open( "travel_time", "rb" ) )
travel_time = pickle.load( open( "sendy_drivers_travel_time", "rb" ) )
geojson = pickle.load( open( "GEOJSON_for_maps", "rb" ) )
df_for_maps = pickle.load( open( "DataFrame_for_maps", "rb" ) )
time_data = pickle.load( open("timedata_for_maps", "rb" ) )
df= pd.merge(df_train, df_riders , on = "Rider Id")
time_to_convert_in_train = ['Arrival at Pickup - Time',
       'Pickup - Time', 'Arrival at Destination - Time']
NAIROBI =(-1.2920659,36.8219462)


for column in time_to_convert_in_train :
    df[column] = df[column].apply(lambda x : pd.to_datetime(x).strftime('%H:%M:%S'))
    df[column] = pd.to_datetime(df[column], format='%H:%M:%S').dt.time
df['Pickup_Hour'] = df['Pickup - Time'].apply(lambda x : x.hour)
df["Speed_in_km/h"] = df["Distance (KM)"] / df['Time from Pickup to Arrival']*3600
def is_week_end(x):
    if x > 5 :
        return 1
    else :
        return 0

df["Week_end"] = df["Placement - Weekday (Mo = 1)"].apply(is_week_end)

#MAP1
cluster_map = "Cluster_Map.html"
map_with_time = "map_with_time.html"

#GRAPH 1
deliveries_per_hour = df.groupby(["Placement - Weekday (Mo = 1)", "Pickup_Hour"]).agg({"Personal or Business": ['count']})
deliveries_per_hour.columns = ['nb_deliveries']
deliveries_per_hour= deliveries_per_hour.reset_index()
deliveries_count = df.groupby(["Placement - Weekday (Mo = 1)","Personal or Business", "Pickup_Hour"]).agg({"Personal or Business": ['count']})
deliveries_count.columns = ['nb_deliveries']
deliveries_count = deliveries_count.reset_index()
deliveries_count = deliveries_count.groupby(["Placement - Weekday (Mo = 1)", 'Personal or Business'])["nb_deliveries"].sum().reset_index()
fig = px.bar(deliveries_count, x="Placement - Weekday (Mo = 1)",
             y="nb_deliveries", color='Personal or Business',
             title = "Sendy activity per week-day"
             , color_discrete_sequence=px.colors.qualitative.Pastel1 )

fig.layout.plot_bgcolor = '#F1F1F1'
fig.layout.paper_bgcolor = '#F1F1F1'

#GRAPH 2
hourly_delivery = px.bar( deliveries_per_hour,
             x='Pickup_Hour',
             y= "nb_deliveries",
             color = "Placement - Weekday (Mo = 1)",
            range_x = [6,23],
            title= "Hourly Pickups")
hourly_delivery.layout.plot_bgcolor = '#F1F1F1'
hourly_delivery.layout.paper_bgcolor = '#F1F1F1'

#MAP 2
uber_cluster_map = "uber_cluster_map.html"



#GRAPH 3
small_speed_df = pd.DataFrame(df_speed.groupby(["sourceid","hod"])["mean_travel_time"].mean()).reset_index()
small_speed_df["mean_travel_time"] = small_speed_df["mean_travel_time"] / 60
uber_speed_per_hour_small = px.scatter(
    small_speed_df,
    x= 'hod',
    y= "mean_travel_time",
    color = "hod" , title= "Hourly Uber Mean Travel time for each cluster")

uber_speed_per_hour_small.layout.plot_bgcolor = '#F1F1F1'
uber_speed_per_hour_small.layout.paper_bgcolor = '#F1F1F1'

#GRAPH 4
img = "week_vs_week_end.png"
encoded_image = base64.b64encode(open(img, 'rb').read())



#GRAPH 5
grouped_multiple = df[df["Speed_in_km/h"] < 50].groupby(['Week_end', 'Pickup_Hour']).agg({'Speed_in_km/h': ['mean']})
grouped_multiple.columns = ['speed_mean']
grouped_multiple = grouped_multiple.reset_index()
sendy_drivers_speed = px.line( grouped_multiple , x='Pickup_Hour', y='speed_mean', color='Week_end')
sendy_drivers_speed.update_layout(title='Sendy\'s drivers speed Analysis')
sendy_drivers_speed.layout.plot_bgcolor = '#F1F1F1'
sendy_drivers_speed.layout.paper_bgcolor = '#F1F1F1'


#GRAPH6
travel_time_outliers = px.scatter(travel_time, x="PickupTime", y="Travel Time", color="Overspeed", title= "Outliers detector" , color_discrete_sequence=px.colors.qualitative.Safe)
travel_time_outliers.layout.plot_bgcolor = '#F1F1F1'
travel_time_outliers.layout.paper_bgcolor = '#F1F1F1'


#MAP3
cluster_map = "14_cluster_map.html"


#MODEL EXPLAINABILITY
model_explainability_1_path = "shap_impact_on_model.png"
model_explainability_1= base64.b64encode(open(model_explainability_1_path, 'rb').read())

model_explainability_2_path = "shap_impact_pos_neg.png"
model_explainability_2 = base64.b64encode(open(model_explainability_2_path, 'rb').read())


app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
server = app.server
app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


app.title = "SENDY LOGISTICS CHALLENGE"





# Add a title
app.layout = html.Div([

    ##HEADER
    html.H1(
      children='🏍️🏍️ SENDY LOGISTIC CHALLENGE 🏍️🏍️',
      style={
         'textAlign': 'center',

      }
    ),

    dcc.Markdown(children= '''
\n
\n
The Sendy Logistic challenge is a Machine Learning project from the Zindi Platform (https://zindi.africa/hackathons/edsa-sendy-logistics-challenge/data) \n
The goal of this challenge is to train a model and predict delivery time of motorbikes drivers in Nairobi, Kenya. This challenge is sponsored by Sendy Logistics. \n
Sendy is a Kenyan logistic company operating in East Africa with a fleet of trucks, cars and motorbikes and ensuring delivery services.
The Machine Learning project will only focus on motorbikes.
Sendy Logistic provided two data sets for this challenge :
The 1st one is composed of data regarding the deliveries, with info like the hour of pickup, the pick-up location / destination, the weather , drivers Id ect..
The 2nd is a data set containing drivers info, like their experience or their rating.
For this challenge, we decided to add some extra data from Uber Movement. (https://movement.uber.com/)
    ''', style={
         'textAlign': 'left',

      }
    ),
    html.Br(),

    html.Br(),
    html.Iframe( id = 'map' , srcDoc = open(map_with_time , "r").read(), width = "90%" , height = '400'
    ),
    dcc.Markdown(children= '''
\n
\n
\n
\n
\n
\n

    ''', style={
         'textAlign': 'left',

      }
    ),


    html.Br(),

    dcc.Tabs([
        dcc.Tab(label=' 📈 DATA EXPLORATION 📉 ', children=[



            dcc.Graph(figure= fig),


            dcc.Graph(figure= hourly_delivery ),
            html.Hr(),


            dcc.Markdown('''
            To provide more info concerning the traffic in Nairobi to our model, we decided to add some info from **Uber**.
            **Uber Movement Dp**t is open sourcing a lot of data on cities. Concerning Nairobi, the city is divided in **107 areas**, represented on the map below.
            For each area, Uber provides **the mean travel time to other areas (per hour for each week day)**. The data  is covering the 3 last years.
            Those data permits to help us to determine rush hour, and have a better overview of the traffic configuration in Nairobi.
            In order to have a better idea of the activity of Sendy Logistics based on the 107 areas of Uber, we add the quantity of picku-ups and deliveries per areas on the map. (You can play with the parameters on the Layer control on the top right of the map).

            '''
            ),

    html.Hr(),
    dcc.Link('See Nairobi in Uber Movement ', href='https://movement.uber.com/explore/nairobi/travel-times/query?lat.=-1.2934283&lng.=36.7760305&z.=11.51&lang=fr-FR&si=81&ti=&ag=sublocations&dt[tpb]=ALL_DAY&dt[wd;]=1,2,3,4,5,6,7&dt[dr][sd]=2019-12-01&dt[dr][ed]=2019-12-31&cd=&sa;=&sdn=&ta;=&tdn='),
    html.Hr(),

    html.Iframe( id = 'map2' , srcDoc = open(uber_cluster_map , "r").read(), width = "85%" , height = '600'
    ),


            dcc.Graph(figure=uber_speed_per_hour_small ),



            # Two columns charts
            html.Div([
                html.Div([html.Hr(),
                    html.H3('WEEK vs WEEK END'),
                     html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))
                ], className="six columns"),
                html.Div([
                    html.H3( "SPEED CALCULATION") ,
                    html.Hr(),
                    dcc.Markdown('''
        With Sendy Logistics data, we could calculate the average speed of each deliveries. This computation permits us to identify that week end is less busy in term of traffic jam and permit to drivers to be faster.
Futhermore, speed calculation is a good way to identify outliers. All red points are considered as overspeed deliveries – in most cases, those outliers came out because the drivers **forgot to confirm pick-up**  so the travel time is really short, but pickup time (time between arrival at pickup and pickup confirmation) is really long. So in order to train our model properly we decided to remove those data (around 2 percente of whole data set)
    ''' )
                ], className="six columns"),
            ], className="row"),


            dcc.Graph(figure=sendy_drivers_speed ),
            html.Hr(),
            dcc.Graph(figure=travel_time_outliers),
            html.Hr(),
            html.Hr(),
            html.H3('CLUSTERING ON GEOGRAPHICAL INFO'),
             dcc.Markdown('''
             In addition to Uber areas, we decided to use a clustering algorithm to subdivide Nairobi in 14 clusters according to the geographic position of pickup and destination.
             ''' ),
             html.Hr(),
             html.Hr(),
            html.Iframe( id = 'map3' , srcDoc = open(cluster_map , "r").read(), width = "70%" , height = '600'),
            html.Hr(),
            html.Hr(),
            html.H3('🛠️ OTHER TOOLS USED 🛠️ '),
             dcc.Markdown('''
             🛎️ Supervised Machine Learning model used : **LIGHT GBM**  \n
                💻 **Hyper Parameters** :  _{ subsample_for_bin= 1262234, reg_lambda = 50, reg_alpha =  7, num_leaves =  18, n_estimators =  37, max_depth =  81, learning_rate =  0.212,random_state = 42, silent = True }_ \n
             🛎️**PCA (Principal Component Analysis) for dimensionality reduction **  on pickup / drop  latititude and longitude \n
             🛎️ Usage of **KNN Imputer** to add several missing values on the temparature in Nairobi during deliveries \n
             🛎️ Adding Rush hour and Week-End features \n
             ''' ),
              html.Hr(),
             html.H3( "🧐🧐 MODEL EXPLAINABILITY 🧐🧐") ,
             dcc.Markdown('''
             Below graphs are showing the features that are mostly impacting the model in its prediction.  \n
             The first one is showing the impact of the top 20 features impacting the model. For example, **the mean travel time** (Uber source) has a weight of 2percent in the final prediction of the modeL. The second graph show the impact of features according to their value. Higher is **the distance, stronger will be the impact of this feature on model's prediction**.
             '''),
             html.Img(src='data:image/png;base64,{}'.format(model_explainability_1.decode())),
              html.Img(src='data:image/png;base64,{}'.format(model_explainability_2.decode()),  style={'height':'70%', 'width':'70%'} ),

              html.Hr()
        ]),



        # Second tab
        dcc.Tab(label=' 🔮🛵 MODEL PREDICTION 🛵🔮 ', children=[
            html.Br(),

            # Two columns charts
            html.Div([



                html.Div([
                    dcc.Markdown(children= '''
                    In this section, you can use the trained model and predict the ETA of your next delivery in Nairobi! \n
                    (If you don't know any places in Nairobi, don't worry, examples are provided below - just copy paste)
                '''
                    ),

               html.Hr(),
                    html.H3('🏭  ➡️ 🏡 '),
                    dcc.Input(
            id="pickup_adress",
            placeholder="🏠 PickUp Adress 🏠 ",value = '🏠 PickUp Adress 🏠'
                 ),
                 dcc.Input(
            id="destination_adress",
            placeholder="📦 Destination Adress 📦 ",value = '📦 Destination Adress 📦'
                 ),

        html.Hr(),
                    html.H3('  🕒 PICK-UP TIME 🕒 '),

                html.Div(
    [
        dcc.Input(id="hour", type="number", placeholder="Hour", min=6, max=23, step=1),
        dcc.Input(
            id="minutes", type="number", debounce=True, placeholder="Minutes", min=0, max=60, step=1
        ),

        html.Hr(),

         html.H3(' 📅 PICK-UP DAY 📅 ' ),
        html.Div(id="number-out"),
    ]
),
html.Div(
    [
        dcc.Input(id="day", type="number", placeholder="Day", min=1, max=31, step=1),
        dcc.Input(
            id="month", type="number",
            debounce=True, placeholder="Month",min=1, max=12, step=1
        ),
        dcc.Input(
            id="year", type="number", placeholder="year",
            min=2020, max=2040, step=1
        ),
        html.Hr(),
        html.Div(id="number-out2"),
        html.Hr()
    ]
),

                    html.Br(),

                    dcc.Markdown(children= '''
                    Nairobi Adresses examples 🇰🇪 🇰🇪  : \n
                    \n
                    🌍 Macharia Rd, Nairobi, Kenya \n
                    🌍 8 Quarry Rd, Nairobi, Kenya \n
                    🌍 Duma Rd, Nairobi, Kenya \n
                    🌍 Kabarnet Ln, Nairobi, Kenya \n
                    🌍 Kilimani, Nairobi, Kenya \n

                     🗺️ Feel free to use other Nairobi locations ! 🗺️


                '''
                    ),

                   html.Br()

                 ], className="six columns"),

                html.Div(children=[
                    html.H3('Delivery info'),
                    html.Br(),
                    html.Br(),


                    html.Button('💡 \n GET YOUR ETA \n 💡 ', id='button', n_clicks=0, style={ 'backgroundColor': "#FF6347 " }),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Div(id='output-state'),
                    html.Br(),
                     dcc.Markdown(children= ''' ⬇️ Please confirm your delivery below ⬇️ ''' ) ,
                    html.Br(),
                    html.Br(),
                    html.Br(),
                    html.Br(),
                ], className="six columns", id='output-container-button'),



                html.Div(children=[
                    html.H3('Confirm Pick-Up'),
                    html.Br(),
                    html.Br(),
                    dcc.Markdown(children= '''
                        To confirm your pick-up please fill your phone number (Your number won't be saved in our system )
                        '''  ),

                    dcc.Markdown(children= '''
                        (Your number won't be saved in our system )
                        '''  ),
                    dcc.Input(id="phone_number", type="number", placeholder="+33698712158"),


                    html.Button(' ✅ CONFIRM ✅ ', id='button_twilio', n_clicks=0, style={ 'backgroundColor': "#FF6347 " }),
                    html.Div(id = 'output-state_twilio'),

                ], className="six columns", id='output-container-button_twilio')
            ], className="row"),


        ])
    ])
], style={'width': '100%', 'textAlign': 'center', 'backgroundColor': "#F1F1F1 " })



@app.callback(Output('output-state', 'children'),
              [Input('button', 'n_clicks')], #n_click
              [State('pickup_adress', 'value'), #input1
               State('destination_adress', 'value'),
               State('hour', 'value'),
               State('minutes', 'value'),
               State('day', 'value'),
               State('month', 'value'),
               State('year', 'value')
               ])

def update_output( n_clicks, input1, input2, input3, input4, input5, input6, input7 ) :
    a = n_clicks
    pickup_location = input1
    destination_location = input2
    pick_up_hour = input3
    pick_up_min = input4
    day = input5
    month = input6
    year = input7

    prediction = predict_time(pickup_location, destination_location,day, month, year,pick_up_hour, pick_up_min )

    return prediction



@app.callback(Output('output-state_twilio', 'children'),
              [Input('button_twilio', 'n_clicks')], #n_click
            [State('phone_number', 'value')]#input2
                )

def send_message_for_confirm(button_twilio, phone_number ) :
    account_sid = 'AC056f8c817dd05f3e07648a05bb18621d'
    auth_token = 'e20fcc39bb7799455ec09abe9c3005ff'
    client = Client(account_sid, auth_token)

    message = client.messages.create(
                              body='Hi there! Thanks for your order on Sendy Logistics platform - We confirm you order.',
                              from_='+12056971552',
                              to="+" + str(phone_number)
                         )
    return type(phone_number) , phone_number


if __name__ == '__main__':

    app.run_server(debug=True)

    app.run_server(debug=True)
