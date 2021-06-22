import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

import pickle
import plotly.graph_objs as go
import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


########### Define your variables ######

tabtitle = 'Game Start to GG'
sourceurl = 'https://www.kaggle.com/xtyscut/video-games-sales-as-at-22-dec-2016csv?select=Video_Games_Sales_as_at_22_Dec_2016.csv'
githublink = 'https://github.com/tzhang900/ds_capstone/tree/main/heroku'
genre_options = ['Sports', 'Platform', 'Racing', 'Role-Playing', 'Puzzle', 'Misc', 'Shooter', 'Simulation', 'Action', 'Fighting', 'Adventure', 'Strategy']
platform_options = ['Wii', 'NES', 'GB', 'DS', 'X360', 'PS3', 'PS2', 'SNES', 'GBA', 'PS4', '3DS', 'N64', 'PS', 'XB', 'PC', '2600', 'PSP', 'XOne', 'WiiU', 'GC', 'GEN', 'DC', 'PSV', 'SAT', 'SCD', 'WS', 'NG', 'TG16', '3DO', 'GG', 'PCFX']
region_options = ['Japan', 'Europe', 'Other']
image1='game.jpg'
image2='consoles.jpg'
background='background.jpg'


### open pickle files
filename = open('model_components/na_eu_lr_model.pkl', 'rb')
lr_eu=pickle.load(filename)
filename.close()

filename = open('model_components/na_jp_lr_model.pkl', 'rb')
lr_jp=pickle.load(filename)
filename.close()

filename = open('model_components/na_other_lr_model.pkl', 'rb')
lr_other=pickle.load(filename)
filename.close()

filename = open('model_components/genre_label.pkl', 'rb')
label_genre=pickle.load(filename)
filename.close()

filename = open('model_components/platform_label.pkl', 'rb')
label_platform=pickle.load(filename)
filename.close()

########### Initiate the app
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=tabtitle


########### Set up the layout

app.layout = html.Div(
    style={
    },
    children=[
        html.H1('Game Sales by Region Prediction'),
        html.Div([
            html.Div([
                    html.H6('Region to Predict:'),
                    dcc.Dropdown(
                        id='regions-drop',
                        options=[{'label': i, 'value': i} for i in region_options],
                        value='Europe'
                    ),
                    html.H6("Copies Sold in North America in millions, eg 3.3"),
                    dcc.Input(id='na-sales', type="number"),
                    html.H6('Select Genre:'),
                    dcc.Dropdown(
                        id='genres-drop',
                        options=[{'label': i, 'value': i} for i in genre_options],
                        value='Sports'
                    ),
                    html.H6('Select Platform:'),
                    dcc.Dropdown(
                        id='platforms-drop',
                        options=[{'label': i, 'value': i} for i in platform_options],
                        value='Wii'
                    ),
                    html.Img(src=app.get_asset_url(image2), style={'width': '100%', 'height': '10%'}),
            ], className='four columns'),
            html.Div([
                html.Div(id='your-output-here', children='',  style={'color': 'blue', 'fontSize': 40}),
                html.Img(src=app.get_asset_url(image1), style={'width': '80%', 'height': '20%'}),
            ], className='eight columns'),
        ], className='twelve columns'),
        html.P(),
        html.Br(),
        html.A('Code on Github', href=githublink),
        html.Br(),
        html.A("Data Source", href=sourceurl),
    ]
)

def listToString(f): 
    
    # initialize an empty string
    str1 = " " 
    s = ['{:.4f}'.format(x) for x in f]
    
    # return string  
    return (str1.join(s))
    
# make a function that can intake any varname and produce a map.
@app.callback(Output('your-output-here', 'children'),
             [Input('regions-drop', 'value'),
              Input('na-sales', 'value'),
              Input('genres-drop', 'value'),
              Input('platforms-drop', 'value')])
def make_figure(region, sales,genre,platform):
    if (sales is None or genre is None or genre == " " or platform is None or platform == " " or region is None):
       return f'Please enter all values to predict the result'
    int_genre = list(label_genre).index(genre)
    int_platform = list(label_platform).index(platform)

    input_value = [[int_genre, int_platform, sales]]
    if region == "Europe":
        prediction = lr_eu.predict(input_value)
    elif region == "Japan":
        prediction = lr_jp.predict(input_value)
    else:
        prediction = lr_other.predict(input_value)
    return f'Predict {platform} {genre} game sales in {region}: ' +   listToString(prediction) + " million"

############ Deploy
if __name__ == '__main__':
    app.run_server(debug=True)
